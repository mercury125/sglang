"""Optional checkpoint donation must not exhaust an admitted request's slots."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import finish_req

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaCacheDonation(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(
                mamba_radix_cache_strategy="extra_buffer",
                _mamba_cache_chunk_size=64,
                disable_radix_cache=False,
                enable_int8_mamba_checkpoint=False,
            )
        )
        self.enterContext(envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(True))

    def make_peak(self, *, lazy=False, spare=0):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=4,
            n_groups=1,
            num_heads=1,
            head_dim=4,
            state_size=4,
            conv_kernel=2,
        )
        pool = HybridReqToTokenPool(
            size=2,
            mamba_size=2 * (3 if lazy else 4) + spare,
            mamba_spec_state_size=2,
            max_context_len=128,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(
                shape=shape,
                layers=[0],
                dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            ),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=lazy,
        )
        allocator = TokenToKVPoolAllocator(
            size=256,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=64,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
                tree_core_backend="python",
            )
        )
        reqs = []
        for i in range(2):
            prefix = array("q", [i + 1] * 64)
            cache.insert(
                InsertParams(
                    key=RadixKey(prefix),
                    value=allocator.alloc(64),
                    mamba_value=pool.mamba_allocator.alloc(1),
                )
            )
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=prefix + array("q", [i + 3] * 64),
                sampling_params=SamplingParams(max_new_tokens=1),
            )
            req.refresh_fill_ids()
            match = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(req.origin_input_ids), cow_mamba=True, req=req
                )
            )
            req.last_node = match.last_device_node
            req.lock = cache.lock(req.last_node)
            pool.alloc([req])
            pool.write(
                (req.kv.req_pool_idx, slice(0, 128)),
                torch.cat(
                    (cache.path_device_indices(req.last_node), allocator.alloc(64))
                ),
            )
            req.kv.cache_protected_len = req.kv.cache_inserted_len = 64
            req.kv.kv_allocated_len = req.kv.kv_committed_len = 128
            req.kv.mamba_last_track_idx = 0
            req.kv.mamba_last_track_seqlen = 128
            req.kv.mamba_prev_track_seqlen = None if lazy else 64
            reqs.append(req)
        self.assertEqual(pool.mamba_allocator.available_size(), spare)
        self.assertEqual(cache.mamba_evictable_size(), 0)
        self.assertEqual(cache.mamba_protected_size(), 2)
        return cache, pool, reqs

    def assert_donated(self, cache, pool, req, old_slot):
        self.assertEqual(req.kv.cache_protected_len, 128)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        self.assertIsNone(req.kv.mamba_prev_track_seqlen)
        value = cache.tree_core.get_component_device_value(
            req.last_node, ComponentType.MAMBA
        )
        self.assertEqual(value.tolist(), [old_slot])
        self.assertNotIn(old_slot, req.kv.mamba_ping_pong_track_buffer.tolist())
        torch.testing.assert_close(
            pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req.kv.req_pool_idx],
            req.kv.mamba_ping_pong_track_buffer,
        )

    def test_locked_pool_preserves_ownership_and_retries(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, pool, reqs = self.make_peak(lazy=lazy)
                req, other = reqs
                old_slot = int(req.kv.mamba_ping_pong_track_buffer[0])
                old_buffer = req.kv.mamba_ping_pong_track_buffer.clone()
                old_row = pool.req_to_token[req.kv.req_pool_idx].clone()
                old_lock, old_node = req.lock, req.last_node
                for _ in range(2):
                    cache.checkpoint(req, up_to=128)
                    self.assertIs(req.lock, old_lock)
                    self.assertEqual(req.last_node, old_node)
                    self.assertEqual(req.kv.cache_protected_len, 64)
                    self.assertEqual(req.kv.cache_inserted_len, 64)
                    self.assertEqual(req.kv.mamba_last_track_seqlen, 128)
                    self.assertEqual(pool.mamba_allocator.available_size(), 0)
                    self.assertEqual(cache.mamba_protected_size(), 2)
                    torch.testing.assert_close(
                        req.kv.mamba_ping_pong_track_buffer, old_buffer
                    )
                    torch.testing.assert_close(
                        pool.req_to_token[req.kv.req_pool_idx], old_row
                    )

                # Finishing transfers a checkpoint without needing a replacement slot.
                finish_req(cache, other, up_to=128)
                cache.checkpoint(req, up_to=128)
                self.assert_donated(cache, pool, req, old_slot)
                finish_req(cache, req, up_to=128)
                cache.evict(
                    EvictParams(num_tokens=256, mamba_num=pool.mamba_allocator.size)
                )
                self.assertEqual(
                    pool.mamba_allocator.available_size(), pool.mamba_allocator.size
                )
                self.assertEqual(cache.token_to_kv_pool_allocator.available_size(), 256)
                self.assertEqual(
                    sorted(pool.mamba_allocator.free_slots.tolist()),
                    list(range(1, pool.mamba_allocator.size + 1)),
                )

    def test_spare_slot_donates_normally(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                cache, pool, reqs = self.make_peak(lazy=lazy, spare=1)
                req = reqs[0]
                old_slot = int(req.kv.mamba_ping_pong_track_buffer[0])
                cache.checkpoint(req, up_to=128)
                self.assert_donated(cache, pool, req, old_slot)
                self.assertEqual(pool.mamba_allocator.available_size(), 0)

    def test_evictable_state_supplies_replacement(self):
        cache, pool, reqs = self.make_peak()
        req, other = reqs
        cache.unlock(other.lock)
        other.lock = cache.lock(
            other.last_node, skip_lock_components=(ComponentType.MAMBA,)
        )
        old_slot = int(req.kv.mamba_ping_pong_track_buffer[0])
        cache.checkpoint(req, up_to=128)
        self.assert_donated(cache, pool, req, old_slot)
        self.assertIsNone(
            cache.tree_core.get_component_device_value(
                other.last_node, ComponentType.MAMBA
            )
        )


if __name__ == "__main__":
    unittest.main()
