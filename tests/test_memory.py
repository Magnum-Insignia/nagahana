"""Memory: access matrix, region enforcement, retention by trigger only, cache compatibility, event log."""

import pytest
import torch

from nagahana.core.errors import AccessDenied, InvariantViolation, ProposalNotEnabled
from nagahana.core.roles import Role
from nagahana.memory.access import Op, Region, allowed, check
from nagahana.memory.eventlog import HashChainLog
from nagahana.memory.kvcache import CACHE_DTYPE, CACHE_ELEMENT_BYTES, KVCacheMeta, assert_compatible
from nagahana.memory.regions import DictStore, MemoryRegion
from nagahana.memory.retention import ConstantRetention


def test_decided_access_rules():
    assert allowed(Role.SIMULATOR, Region.ENVIRONMENT) == Op.READ | Op.WRITE
    assert allowed(Role.ADVISOR, Region.ENVIRONMENT) == Op.READ          # D-01 decided
    assert allowed(Role.ADVISOR, Region.IMAGINATION) == Op.READ
    assert allowed(Role.FORECASTER, Region.ENVIRONMENT) == Op.READ       # beliefs never written to facts
    assert allowed(Role.GENERATOR, Region.ENVIRONMENT) == Op.NONE
    with pytest.raises(AccessDenied):
        check(Role.FORECASTER, Region.ENVIRONMENT, Op.WRITE)
    with pytest.raises(AccessDenied):
        check(Role.ADVISOR, Region.IMAGINATION, Op.WRITE)                # memory-less


def test_decoder_reads_monitor_only_under_proposal():
    assert allowed(Role.DECODER, Region.MONITOR) == Op.NONE
    assert allowed(Role.DECODER, Region.MONITOR, ("P-12",)) == Op.READ


def test_region_enforces_access():
    env = MemoryRegion(Region.ENVIRONMENT, DictStore())
    env.write(Role.SIMULATOR, "k", 1)
    assert env.read(Role.ADVISOR, "k") == 1
    with pytest.raises(AccessDenied):
        env.write(Role.FORECASTER, "k", 2)


def test_retention_depends_on_trigger_index_only():
    r = ConstantRetention(0.1)
    assert r.alpha(0) == r.alpha(10_000) == 0.1
    with pytest.raises(InvariantViolation):
        ConstantRetention(1.0)


def test_kv_cache_refuses_other_weights():
    meta = KVCacheMeta(Region.ENVIRONMENT, "tstct", "a" * 64, "z-v0", "0.1", 2, 4, 16)
    assert meta.bytes_per_position(2) == 2 * 2 * 4 * 16 * 2
    # D-54: caches are stored in fp32, so the default element size is 4 bytes.
    assert CACHE_DTYPE is torch.float32 and CACHE_ELEMENT_BYTES == torch.finfo(CACHE_DTYPE).bits // 8 == 4
    assert meta.bytes_per_position() == 2 * 2 * 4 * 16 * 4
    assert_compatible(meta, model_hash="a" * 64, latent_space="z-v0")
    with pytest.raises(InvariantViolation):
        assert_compatible(meta, model_hash="b" * 64, latent_space="z-v0")


def test_event_log_is_opt_in_and_tamper_evident():
    with pytest.raises(ProposalNotEnabled):
        HashChainLog(enabled_proposals=())
    log = HashChainLog(enabled_proposals=("P-02",))
    for i in range(3):
        log.append(f"update-{i}".encode())
    log.verify()
    entries = log.entries()
    tampered = entries[1].__class__(1, entries[1].prev_hash, entries[1].payload_hash, entries[1].entry_hash, b"edited")
    log._entries[1] = tampered
    with pytest.raises(InvariantViolation):
        log.verify()
