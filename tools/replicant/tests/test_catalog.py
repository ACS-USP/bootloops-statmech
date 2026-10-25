"""Catalog: entries, routes, and the banked numbers each entry reproduces.

Conventions: every catalog entry runs its level-(b) family pipeline (or its cavity
pipeline) through core.run_pipeline; a RefCheck names a banked key in
reference/<file>.json, the run parameters at which the entry reproduces it, and how the
compared number is read off the run's values.  Published numbers are never retyped
here: the comparison is refdata.check against the bank.  Statistical (population-
dynamics) keys are compared in a 99% Student-t band (cavity/_stats.t_quantile) on
hypot(stderr, banked tol), in the slow tier only.
"""
import json
import os
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from replicant.catalog import UNCLAIMED, RefCheck, catalog, check_ref
from replicant.testing import slow

BANK = Path(__file__).resolve().parent.parent / "reference"

ROUTES = {"rem": {"replica"}, "sk": {"replica", "cavity", "amp"}, "pspin": {"replica"},
          "ising_pspin": {"replica"}, "xorsat": {"cavity"}, "ksat": {"cavity"}, "ldpc_bec": {"cavity"},
          "spiked_wigner": {"amp"}, "sbm": {"cavity"}, "ridge": {"replica"},
          "perceptron": {"replica", "cavity"}, "teacher_student": {"replica", "amp"},
          "hopfield": {"replica", "cavity"}, "viana_bray": {"cavity"}}


def _bank_keys():
    return {(p.stem, k) for p in BANK.glob("*.json") for k in json.loads(p.read_text())}


def test_catalog_lists_the_v1_entries_with_their_routes():
    cat = catalog()
    assert {n: e.routes for n, e in cat.items()} == ROUTES
    for name, e in cat.items():
        assert e.name == name and e.cite and e.spec.family and e.refs, name
        for route in e.routes:
            steps = e.route_steps[route]
            assert steps and all(s.cite for s in steps), (name, route)


def test_catalog_entry_reproduces_its_banked_numbers():
    """Default tier (ruling R5): every non-slow, non-statistical key of every entry."""
    before = mp.mp.dps
    for name, e in catalog().items():
        fast = [r for r in e.refs if not (r.slow or r.statistical)]
        assert fast, f"{name}: no default-tier banked check"
        for ref in fast:
            out = check_ref(e, ref)
            assert out.ok, (name, ref.key, out)
    assert mp.mp.dps == before


def test_check_ref_can_fail():
    """Positive control: the comparison is not vacuous (wrong quantity, wrong point)."""
    sk = catalog()["sk"]
    assert not check_ref(sk, RefCheck("sk", "tc", lambda v: v["rs_ground_energy"])).ok
    rem = catalog()["rem"]
    wrong = RefCheck("rem", "ground_energy_derrida", lambda v: v["f"], params={"beta": 1})
    assert not check_ref(rem, wrong).ok          # paramagnet at beta = 1: not the ground energy


@slow
def test_catalog_entry_reproduces_its_slow_and_statistical_numbers():
    for name, e in catalog().items():
        for ref in e.refs:
            if ref.slow or ref.statistical:
                out = check_ref(e, ref)
                assert out.ok, (name, ref.key, out)


def test_every_ref_key_exists_in_the_bank():
    keys = _bank_keys()
    for name, e in catalog().items():
        for ref in e.refs:
            assert (ref.file, ref.key) in keys, (name, ref.file, ref.key)
            assert ref.route in e.routes, (name, ref.key, ref.route)
            if ref.statistical:
                assert ref.slow and ref.error is not None, (name, ref.key)


def test_unclaimed_bank_keys_are_listed():
    """Guard against bank rot: every banked key is claimed by an entry or allowlisted."""
    keys = _bank_keys()
    claimed = {(r.file, r.key) for e in catalog().values() for r in e.refs}
    unclaimed = set(UNCLAIMED)
    assert keys == claimed | unclaimed, (keys - claimed - unclaimed, (claimed | unclaimed) - keys)
    assert not claimed & unclaimed, claimed & unclaimed
    assert all(isinstance(v, str) and len(v) > 20 for v in UNCLAIMED.values())
