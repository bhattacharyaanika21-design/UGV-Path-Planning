"""Fast closed-loop smoke tests (about 30 s in total). Run with:  python -m pytest -q"""
import filecmp
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from av.scenarios import SCENARIOS  # noqa: E402
from av.sim import run  # noqa: E402


@pytest.mark.parametrize("key", ["S5", "S6"])
def test_scenario_completes_without_collision(key):
    m = run(SCENARIOS[key](0), seed=0, record=False)["metrics"]
    assert not m["collision"], m["collision_info"]
    assert m["completed"]


def test_every_scenario_builds():
    for key, build in SCENARIOS.items():
        spec = build(0)
        assert spec.world.agents, key
        assert len(spec.route) >= 2, key


def test_ros_package_uses_identical_av_library():
    """ros2_ws/src/bharat_sim/bharat_sim/av must stay in sync (tools/sync_av.sh)."""
    a = os.path.join(ROOT, "av")
    b = os.path.join(ROOT, "ros2_ws", "src", "bharat_sim", "bharat_sim", "av")
    names = [f for f in os.listdir(a) if f.endswith(".py")]
    match, mismatch, errors = filecmp.cmpfiles(a, b, names, shallow=False)
    assert not mismatch and not errors, f"out of sync: {mismatch + errors}"
