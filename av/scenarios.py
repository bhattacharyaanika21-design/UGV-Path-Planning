"""Six Indian road scenarios (five required + one bonus).

Each builder takes a seed; the seed jitters speeds, trigger distances,
positions and wander noise so Monte-Carlo runs give a completion rate
rather than one hand-tuned pass.

 S1 village   – unmarked, narrow, curvy village road: potholes, bicycle,
                pedestrians on the edge, parked bullock cart, oncoming bus
                and tractor, overtaking two-wheeler, roadside cattle.
 S2 junction  – busy unsignalised urban 4-way, right turn across oncoming
                traffic, diagonal two-wheeler, wrong-way rider, pedestrians
                crossing anywhere, stopped bus hiding a pedestrian.
 S3 highway   – highway with slow tractor/truck, bus merging from the slip
                road without signalling, fast overtaker, wrong-way rider on
                the shoulder, stalled car or pedestrian crossing.
 S4 market    – dense market lane with wandering pedestrians, pushcarts,
                parked autos, a cow, an auto stopping abruptly; a delivery
                truck + pushcart block the lane ahead -> global reroute.
 S5 cattle    – rural road, herd emerges from behind bushes, one stops
                mid-road, one turns back; oncoming truck, following car.
 S6 dart-out  – child runs out from in front of a parked bus.
"""
from __future__ import annotations

import numpy as np

from .sim import ScenarioSpec
from .world import Agent, Crosser, PathFollow, RoadNetwork, Wander, World


class _Builder:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.agents = []
        self.seed = seed

    def j(self, x, pct=0.1):
        return float(x * (1 + self.rng.uniform(-pct, pct)))

    def jd(self, x, amt=0.5):
        return float(x + self.rng.uniform(-amt, amt))

    def add(self, cls, behavior=None, x=0.0, y=0.0, yaw=0.0, v=0.0):
        a = Agent(len(self.agents) + 1, cls, x, y, yaw, v, behavior)
        self.agents.append(a)
        return a

    def follow(self, cls, pts, v, s0, d_off, wander=0.0, reverse=False, events=None, idm=True):
        b = PathFollow(pts, v, d_off=d_off, wander=wander, s0=s0, reverse=reverse,
                       events=events, idm=idm, seed=int(self.rng.integers(1e9)))
        return self.add(cls, b, v=v)

    def static(self, cls, x, y, yaw=0.0):
        return self.add(cls, None, x, y, yaw, 0.0)

    def cross(self, cls, pts, speed, trigger, **kw):
        b = Crosser(pts, speed, trigger, seed=int(self.rng.integers(1e9)), **kw)
        return self.add(cls, b)

    def wander(self, cls, c, r, speed):
        b = Wander(c, r, speed, seed=int(self.rng.integers(1e9)))
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        return self.add(cls, b, c[0], c[1], yaw, speed)


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


# ---------------------------------------------------------------------------
def s1_village(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    xs = np.arange(0, 281, 20.0)
    ys = 7.0 * np.sin(xs / 45.0) + 3.0 * np.sin(xs / 17.0)
    names = [f"V{i}" for i in range(len(xs))]
    for n, x, y in zip(names, xs, ys):
        net.add_node(n, x, y)
    for u, v in zip(names[:-1], names[1:]):
        net.add_edge(u, v, width=7.5, speed=11.0)
    ref = net.route_path(names)
    P = ref.xy
    L = ref.length
    c = lambda s, d: tuple(float(q) for q in ref.to_cart(s, d)[:2])

    B.follow("bicycle", P[: int(112 / 0.5)], B.j(4.0), B.jd(55, 5), 2.3, wander=0.25)
    B.follow("pedestrian", P, B.j(1.2), B.jd(95, 4), 3.4, wander=0.06, idm=False)
    B.follow("pedestrian", P, B.j(1.0), L - B.jd(150, 5), 3.4, wander=0.06, reverse=True, idm=False)
    x, y = c(125, 2.7)
    _, _, th, _ = ref.interp(125)
    B.static("bullockcart", x, y, float(th))
    B.follow("bus", P, B.j(8.0), L - B.jd(235, 5), 1.5, wander=0.05, reverse=True)
    B.follow("twowheeler", P[: int(105 / 0.5)], B.j(12.5), 0.0, -0.9, wander=0.3,
             events=[dict(trigger=("time", 4.0), d_off=1.2, lat_speed=0.8)])
    x, y = c(185, 4.6)
    B.wander("cattle", (x, y), 0.4, 0.15)
    potholes = []
    for s_, d_, r_ in [(62, 1.6, 0.55), (65, 0.4, 0.45), (68, 2.4, 0.4), (150, -0.4, 0.6),
                       (205, 1.9, 0.5), (212, 0.6, 0.45)]:
        x, y = c(B.jd(s_, 1.0), B.jd(d_, 0.2))
        potholes.append((x, y, r_))
    decor = [(rect(-10, -60, 290, 60), "#dfe8c9")]
    world = World(net, B.agents, potholes=potholes, decor=decor)
    x0, y0, yaw0 = (float(q) for q in ref.to_cart(14, 1.75))
    return ScenarioSpec("S1_village", "Unmarked village road",
                        "Narrow curvy road, no markings, potholes, oncoming bus beside a parked "
                        "bullock cart, bicycle & pedestrians on the edge, overtaking two-wheeler.",
                        world, names, (x0, y0, float(yaw0), 6.0), zone_speed=11.0,
                        t_max=90.0, view=36)


# ---------------------------------------------------------------------------
def s2_junction(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    for n, x, y in [("W", -95, 0), ("C", 0, 0), ("E", 95, 0), ("N", 0, 95), ("S", 0, -95)]:
        net.add_node(n, x, y)
    for u in ("W", "E", "N", "S"):
        net.add_edge(u, "C", width=12.0, speed=10.0)
    WE = [(-95, 0), (95, 0)]
    EW = [(95, 0), (-95, 0)]
    SN = [(0, -95), (0, 95)]
    NS = [(0, 95), (0, -95)]
    # oncoming traffic (westbound, on their left = south side)
    B.follow("car", EW, B.j(9.0), B.jd(35, 5), 3.0, wander=0.2)
    B.follow("auto", EW, B.j(7.0), B.jd(0, 3), 2.6, wander=0.35)
    B.follow("bus", [(150, 0), (-95, 0)], B.j(7.5), 0.0, 3.0, wander=0.1)
    # cross traffic
    B.follow("car", SN, B.j(8.0), B.jd(30, 5), 3.0, wander=0.2)
    B.follow("auto", NS, B.j(6.0), B.jd(40, 5), 3.0, wander=0.3)
    B.follow("twowheeler", NS, B.j(8.0), B.jd(10, 5), 2.0, wander=0.5)
    # wrong-way rider on our side
    B.follow("twowheeler", EW, B.j(6.0), B.jd(70, 5), -4.6, wander=0.2)
    # diagonal two-wheeler cutting across the junction
    B.cross("twowheeler", [(-22, -13), (0, 0), (24, 14)], B.j(6.5), ("ego_dist", B.j(40)))
    # pedestrians crossing anywhere
    B.cross("pedestrian", [(-38, 7.8), (-38, -7.8)], B.j(1.3), ("ego_dist", B.j(26)))
    B.cross("pedestrian", [(-37, 8.3), (-36.5, -7.5)], B.j(1.1), ("ego_dist", B.j(25)))
    B.cross("pedestrian", [(8, -20), (-8, -20.5)], B.j(1.3), ("ego_dist", B.j(28)),
            pauses=[(0.5, 2.0)])
    # stopped bus at left edge with a pedestrian stepping out in front of it
    B.static("bus", -58, 4.6, 0.0)
    B.cross("pedestrian", [(-51.6, 7.2), (-51.6, 2.2), (-52.5, -7.5)], B.j(1.5),
            ("ego_dist", B.j(19, 0.05)), pauses=[(0.3, 0.6)])
    # street vendor on the exit road
    B.static("pushcart", 4.6, -42, np.pi / 2)
    B.wander("pedestrian", (-9, 10), 1.5, 0.6)
    B.wander("pedestrian", (9, -10), 1.5, 0.6)
    occ = [rect(-95, 9, -9, 40), rect(9, 9, 95, 40), rect(-95, -40, -9, -9), rect(9, -40, 95, -9)]
    decor = [(p, "#c9c3b8") for p in occ]
    world = World(net, B.agents, occluders=occ, decor=decor)
    return ScenarioSpec("S2_junction", "Unsignalised urban intersection",
                        "Right turn across oncoming traffic at a busy 4-way with no signals: "
                        "diagonal & wrong-way two-wheelers, jaywalkers, pedestrian hidden by a bus.",
                        world, ["W", "C", "S"], (-88, 3.0, 0.0, 7.0), zone_speed=10.0,
                        junctions=["C"], junction_cap=4.5, t_max=60.0, view=40)


# ---------------------------------------------------------------------------
def s3_highway(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    net.add_node("H0", 0, 0)
    net.add_node("H1", 520, 0)
    net.add_edge("H0", "H1", width=11.0, speed=19.0)
    main = [(0, 0), (520, 0)]
    ramp = [(20, 34), (100, 18), (160, 7.5), (200, 3.6), (520, 3.6)]
    B.follow("tractor", main, B.j(7.0), B.jd(95, 5), 3.6, wander=0.2)
    B.follow("truck", main, B.j(10.0), B.jd(150, 6), 0.3, wander=0.3)
    B.follow("bus", ramp, B.j(13.0), B.jd(35, 6), 0.0, wander=0.1,
             events=[dict(trigger=("ego_x", B.j(150, 0.05)), d_off=-3.2, lat_speed=0.9)])
    B.follow("car", [(-120, 0), (520, 0)], B.j(25.0), 60.0, -3.6, wander=0.25)
    B.follow("twowheeler", main, B.j(7.0), 520 - B.jd(360, 10), -4.9, reverse=True, wander=0.15)
    B.static("car", 330, 4.3, 0.05)
    B.cross("pedestrian", [(420, 7.5), (420, -7.5)], B.j(1.4), ("ego_dist", B.j(70)))
    decor = [(np.array([[20, 36.5], [100, 20.5], [160, 10], [200, 5.5], [200, 3.0],
                        [160, 5.0], [100, 15.5], [20, 31.5]]), "#9a9a9a"),
             (rect(-10, -9, 530, -6.5), "#6a8a4a")]
    world = World(net, B.agents, decor=decor)
    return ScenarioSpec("S3_highway", "Highway merge with slow vehicles",
                        "Slow tractor & truck, bus merging from the slip road without signalling, "
                        "fast overtaker, wrong-way rider on the shoulder, stalled car, pedestrian.",
                        world, ["H0", "H1"], (10, 0.0, 0.0, 15.0), zone_speed=19.0,
                        keep_left=False, d_pref=0.0, t_max=50.0, view=55)


# ---------------------------------------------------------------------------
def s4_market(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    for n, x, y in [("M0", 0, 0), ("M1", 85, 0), ("M2", 165, 0), ("M3", 240, 0),
                    ("P1", 85, 45), ("P2", 165, 45)]:
        net.add_node(n, x, y)
    for u, v, w in [("M0", "M1", 7.5), ("M1", "M2", 7.5), ("M2", "M3", 7.5),
                    ("M1", "P1", 6.5), ("P1", "P2", 6.5), ("P2", "M2", 6.5)]:
        net.add_edge(u, v, width=w, speed=5.5)
    main = [(0, 0), (240, 0)]
    # --- the blockage (delivery truck + pushcart + parked scooter) ---------
    B.static("truck", 122, 1.35, 0.0)
    B.static("pushcart", 123, -2.1, 0.0)
    B.static("twowheeler", 116, -2.9, 0.3)
    # --- market crowd on main lane -----------------------------------------
    for cx, cy in [(16, 3.9), (30, -3.9), (44, 3.9), (58, -3.9)]:
        B.wander("pedestrian", (B.jd(cx, 2), cy), 0.6, B.j(0.6))
    B.cross("pedestrian", [(31, -4.2), (31, 4.2)], B.j(1.2), ("ego_dist", B.j(13)))
    B.follow("pushcart", main, B.j(0.9), B.jd(22, 2), -2.0, wander=0.1)
    B.static("auto", 38, -2.9, 0.1)
    B.wander("cattle", (B.jd(49, 1), -2.0), 0.3, 0.1)
    B.follow("auto", [(0, 0), (60, 0)], B.j(5.0), B.jd(13, 1), 1.4, wander=0.2,
             events=[dict(trigger=("time", B.j(4.5)), v_des=0.0, duration=B.j(4.0))])
    B.follow("twowheeler", main, B.j(6.0), 0.0, -1.7, wander=0.5)
    # --- side lanes (detour) -------------------------------------------------
    side = [(85, 0), (85, 45), (165, 45), (165, 0)]
    for cx, cy in [(82.5, 25), (110, 47.6), (135, 42.4), (150, 47.5), (167.5, 20)]:
        B.wander("pedestrian", (cx, cy), 0.9, B.j(0.6))
    B.static("car", 120, 47.4, 0.0)
    B.follow("twowheeler", side[::-1], B.j(5.0), B.jd(40, 5), 1.8, wander=0.3)
    B.cross("cattle", [(143, 50), (143, 40)], B.j(0.7), ("ego_dist", B.j(22)),
            pauses=[(0.45, 3.0)])
    B.wander("pedestrian", (200, 3.9), 0.6, 0.6)
    B.static("pushcart", 210, -3.0, 0.0)
    # --- buildings (occlusion) ----------------------------------------------
    occ = [rect(-5, 5.5, 79, 30), rect(91, 5.5, 159, 39.5), rect(171, 5.5, 245, 30),
           rect(-5, -30, 245, -5.5), rect(-5, 51.5, 245, 70), rect(20, 30, 79, 70),
           rect(171, 30, 245, 70)]
    decor = [(p, "#d8c7a8") for p in occ]
    world = World(net, B.agents, occluders=occ, decor=decor)
    return ScenarioSpec("S4_market", "Dense market + blocked lane (reroute)",
                        "Crowded market lane; a delivery truck and pushcart block the lane ahead, "
                        "so the vehicle reroutes through the parallel street.",
                        world, ["M0", "M1", "M2", "M3"], (6, 1.2, 0.0, 3.0), zone_speed=5.5,
                        junctions=["M1", "P1", "P2", "M2"], junction_cap=3.5, t_max=200.0, view=34)


# ---------------------------------------------------------------------------
def s5_cattle(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    net.add_node("R0", 0, 0)
    net.add_node("R1", 320, 0)
    net.add_edge("R0", "R1", width=8.0, speed=15.0)
    main = [(0, 0), (320, 0)]
    B.wander("cattle", (70, 5.3), 0.4, 0.12)
    B.follow("car", main, 14.0, 5.0, 2.0, wander=0.1)
    B.follow("truck", main, B.j(11.0), 320 - B.jd(270, 8), 2.0, reverse=True, wander=0.1)
    trig = B.j(60, 0.08)
    specs = [(0.9, None, None), (0.85, [(0.45, 5.0)], None), (1.0, None, None),
             (0.8, None, 0.35), (0.9, None, None)]
    for i, (sp, pauses, tb) in enumerate(specs):
        x0 = 146 + 3.2 * i + B.jd(0, 0.6)
        B.cross("cattle", [(x0, 5.4 + 0.3 * (i % 2)), (x0 - 1.5 + B.jd(0, 1), -12)],
                B.j(sp), ("ego_dist", trig + 3.5 * i), pauses=pauses, turn_back=tb)
    occ = [rect(125, 4.9, 165, 16), rect(172, -4.9, 205, -16)]
    decor = [(rect(-10, -40, 330, -4.2), "#dfe8c9"), (rect(-10, 4.2, 330, 40), "#dfe8c9")] + \
        [(p, "#5e8c4a") for p in occ]
    world = World(net, B.agents, occluders=occ, decor=decor)
    return ScenarioSpec("S5_cattle", "Sudden cattle crossing",
                        "A herd walks out from behind roadside bushes; one animal stops mid-road "
                        "and one turns back, with an oncoming truck and a car following close behind.",
                        world, ["R0", "R1"], (25, 2.0, 0.0, 14.0), zone_speed=15.0,
                        t_max=60.0, view=45)


# ---------------------------------------------------------------------------
def s6_dartout(seed=0):
    B = _Builder(seed)
    net = RoadNetwork()
    net.add_node("A", 0, 0)
    net.add_node("B", 210, 0)
    net.add_edge("A", "B", width=10.0, speed=11.0)
    B.static("bus", 95, 3.65, 0.0)
    B.cross("pedestrian", [(101.4, 6.2), (101.4, 2.0), (101.0, -6.5)], B.j(2.2),
            ("ego_dist", B.j(24, 0.05)))
    B.follow("auto", [(210, 0), (0, 0)], B.j(7.0), B.jd(70, 5), 2.4, wander=0.2)
    B.static("car", 60, -4.0, 0.0)
    B.static("car", 140, -4.0, 0.0)
    occ = [rect(-5, 7.6, 215, 20), rect(-5, -7.6, 215, -20)]
    decor = [(p, "#d8c7a8") for p in occ]
    world = World(net, B.agents, occluders=occ, decor=decor)
    return ScenarioSpec("S6_dartout", "Occluded pedestrian dart-out (bonus)",
                        "A child runs out from in front of a stopped bus while an auto approaches "
                        "from the other side.", world, ["A", "B"], (5, 2.5, 0.0, 10.0),
                        zone_speed=11.0, t_max=40.0, view=36)


SCENARIOS = dict(S1=s1_village, S2=s2_junction, S3=s3_highway, S4=s4_market, S5=s5_cattle,
                 S6=s6_dartout)
