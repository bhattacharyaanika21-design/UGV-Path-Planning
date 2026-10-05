#!/usr/bin/env python3
"""Export every scenario (road network, all actor trajectories, ego trajectory)
to CSV/JSON so it can be replayed in MATLAB's drivingScenario / Driving
Scenario Designer, or used to author the matching RoadRunner scenes.

    python3 export_matlab.py          # writes matlab/data/<scenario>/...
Then in MATLAB:
    cd matlab; replay_scenario("S1_village")
"""
from __future__ import annotations

import csv
import json
import os

from av.scenarios import SCENARIOS
from av.sim import run


def export(key, seed=0, root="matlab/data"):
    spec = SCENARIOS[key](seed)
    res = run(spec, seed=seed, record=True)
    name = res["metrics"]["scenario"]
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    net = spec.world.network
    road = dict(
        title=spec.title, description=spec.description,
        nodes={k: [float(v[0]), float(v[1])] for k, v in net.nodes.items()},
        edges=[dict(u=k[0], v=k[1], width=e["width"], speed=e["speed"],
                    pts=[[float(p[0]), float(p[1])] for p in e["pts"]])
               for k, e in net.edges.items()],
        route_initial=spec.route, route_final=res["route"],
        potholes=[list(map(float, p)) for p in spec.world.potholes],
        occluders=[[list(map(float, q)) for q in poly] for poly in spec.world.occluders],
    )
    with open(os.path.join(d, "road.json"), "w") as f:
        json.dump(road, f, indent=1)
    with open(os.path.join(d, "actors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "id", "class", "x", "y", "yaw", "v", "length", "width"])
        for fr in res["frames"]:
            for (aid, cls, x, y, yaw, v, L, W) in fr["actors"]:
                w.writerow([f"{fr['t']:.2f}", aid, cls, f"{x:.3f}", f"{y:.3f}", f"{yaw:.4f}",
                            f"{v:.3f}", L, W])
    L = res["log"]
    with open(os.path.join(d, "ego.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "v", "a", "steer", "state"])
        for i in range(0, len(L["t"]), 2):
            w.writerow([f"{L['t'][i]:.2f}", f"{L['x'][i]:.3f}", f"{L['y'][i]:.3f}",
                        f"{L['yaw'][i]:.4f}", f"{L['v'][i]:.3f}", f"{L['a'][i]:.3f}",
                        f"{L['delta'][i]:.4f}", L["state"][i]])
    print(f"exported {name} -> {d}")


if __name__ == "__main__":
    for k in SCENARIOS:
        export(k)
