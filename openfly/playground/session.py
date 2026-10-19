"""The live simulation behind the playground.

A :class:`Session` owns one brain, one :class:`~openfly.Simulator` and
(optionally) one :class:`~openfly.agent.VirtualFly`. A background thread
advances the simulation in small chunks (20 ms of brain time) and pushes
a *frame* after each chunk to every connected browser:

* which neurons spiked (compact base64 array),
* firing rate per super class,
* behavioural drives (feeding, walking, turning, escape...),
* the active stimuli, and in fly mode the fly's position and state.

Commands from the browser (add a smell, silence a cell type, switch to
the virtual fly...) are applied between chunks, under a lock.
"""

from __future__ import annotations

import base64
import queue
import threading
import time
import traceback

import numpy as np

from .. import motor
from ..senses import mechano, olfaction, scenarios, thermo, vision
from ..senses.base import Stimulus, as_stimuli
from ..senses.taste import taste as taste_fn
from ..sim.simulator import Simulator
from . import lessons

PALETTE_CLASS = {
    "optic": "#3fa7ff", "central": "#ff7ab6", "sensory": "#ffd166", "visual_projection": "#06d6a0",
    "ascending": "#c77dff", "descending": "#ff5d5d", "sensory_ascending": "#f4a261", "visual_centrifugal": "#4cc9f0",
    "motor": "#ff9f1c", "endocrine": "#b5e48c", "": "#7f8fa6",
}
PALETTE_NT = {
    "acetylcholine": "#ffb703", "gaba": "#4361ee", "glutamate": "#9d4edd", "dopamine": "#f72585",
    "serotonin": "#4cc9f0", "octopamine": "#80ed99", "histamine": "#fb8500", "": "#7f8fa6",
}
PALETTE_SIDE = {"left": "#ff6b6b", "right": "#4ecdc4", "center": "#ffe66d", "": "#7f8fa6"}
_EXTRA = ["#e9c46a", "#2a9d8f", "#e76f51", "#8ecae6", "#bde0fe", "#ffafcc", "#cdb4db", "#a8dadc"]


def _codes(values: np.ndarray, palette: dict):
    names = list(dict.fromkeys(list(palette) + sorted(set(values) - set(palette))))
    lookup = {n: k for k, n in enumerate(names)}
    codes = np.array([lookup[v] for v in values], dtype=np.uint8)
    colors = [palette.get(n, _EXTRA[k % len(_EXTRA)]) for k, n in enumerate(names)]
    return codes, names, colors


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


class Session:
    def __init__(self, brain, chunk_ms: float = 20.0, seed: int = 0):
        self.brain = brain
        self.chunk_ms = chunk_ms
        self.sim = Simulator(brain, seed=seed)
        self.motor = motor.MotorMonitor(brain)
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue] = []
        self.stimuli: dict[int, dict] = {}
        self._next_id = 1
        self.running = True
        self.speed = 1.0
        self.mode = "brain"
        self.fly = None
        self.arena_name = None
        self.jobs: dict[int, dict] = {}
        self._job_id = 1
        self.frame_no = 0
        self.wall_per_chunk = 0.0
        n = brain.neurons
        self.class_codes, self.class_names, self.class_colors = _codes(n["super_class"].to_numpy(), PALETTE_CLASS)
        self.nt_codes, self.nt_names, self.nt_colors = _codes(n["nt"].to_numpy(), PALETTE_NT)
        self.side_codes, self.side_names, self.side_colors = _codes(n["side"].to_numpy(), PALETTE_SIDE)
        self.class_sizes = np.bincount(self.class_codes, minlength=len(self.class_names))
        self._stop = False
        self.thread = threading.Thread(target=self._loop, name="openfly-session", daemon=True)

    # ---------------------------------------------------------- plumbing
    def start(self) -> "Session":
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop = True

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=8)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast(self, frame: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

    # -------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop:
            if not self.running:
                time.sleep(0.05)
                continue
            t0 = time.time()
            try:
                with self.lock:
                    frame = self._tick()
            except Exception:  # never let the loop die; report instead
                frame = {"type": "error", "message": traceback.format_exc(limit=3)}
                self.running = False
            self.wall_per_chunk = 0.9 * self.wall_per_chunk + 0.1 * (time.time() - t0)
            self._broadcast(frame)
            budget = self.chunk_ms / 1000.0 / max(self.speed, 0.05)
            time.sleep(max(0.0, budget - (time.time() - t0)))

    def _tick(self) -> dict:
        fly_state = None
        if self.mode == "fly" and self.fly is not None:
            self.fly.extra_stimuli = self._all_stimuli()  # user stimuli on top of the world
            rec = self.fly.step()
            neurons = self.fly.last_spikes
            drive = self.fly.last_drive
            fly_state = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}
            fly_state["events"] = self.fly.events[-5:]
            fly_state["items"] = [{"kind": it.kind, "x": getattr(it, "x", None), "y": getattr(it, "y", None),
                                   "name": it.name, "fired": getattr(it, "fired", None)} for it in self.fly.arena.items]
        else:
            self.sim.stimuli = self._all_stimuli()
            _, neurons = self.sim.advance(self.chunk_ms)
            drive = self.motor.update(neurons, self.chunk_ms)
        self._expire_stimuli()
        uniq = np.unique(neurons).astype(np.uint32)
        counts = np.bincount(self.class_codes[uniq], minlength=len(self.class_names)) if uniq.size else np.zeros(len(self.class_names))
        spikes_per_class = np.bincount(self.class_codes[neurons], minlength=len(self.class_names)) if neurons.size else counts
        rates = spikes_per_class / np.maximum(self.class_sizes, 1) / (self.chunk_ms / 1000.0)
        self.frame_no += 1
        return {
            "type": "frame",
            "frame": self.frame_no,
            "t": round(self.engine_time(), 2),
            "chunk_ms": self.chunk_ms,
            "n_spikes": int(neurons.size),
            "n_active": int(uniq.size),
            "awake": self.sim.engine.n_awake if self.mode == "brain" else self.fly.sim.engine.n_awake,
            "spikes": b64(uniq),
            "class_rates": {self.class_names[k]: round(float(r), 3) for k, r in enumerate(rates) if self.class_sizes[k]},
            "drive": {k: round(float(v), 3) for k, v in drive.items()},
            "stimuli": self.stimulus_list(),
            "fly": fly_state,
            "mode": self.mode,
            "wall_ms": round(self.wall_per_chunk * 1000, 1),
            "running": self.running,
        }

    def engine_time(self) -> float:
        if self.mode == "fly" and self.fly is not None:
            return self.fly.sim.engine.time_ms
        return self.sim.engine.time_ms

    # --------------------------------------------------------- stimuli
    def _all_stimuli(self) -> list:
        out = []
        for entry in self.stimuli.values():
            out.extend(entry["stims"])
        return out

    def _expire_stimuli(self) -> None:
        t = self.engine_time()
        dead = [k for k, e in self.stimuli.items() if e["stims"] and all(s.stop_ms is not None and s.stop_ms <= t for s in e["stims"])]
        for k in dead:
            del self.stimuli[k]

    def stimulus_list(self) -> list:
        return [{"id": k, "label": e["label"], "kind": e["kind"], "n": e["n"]} for k, e in self.stimuli.items()]

    def add_stimulus(self, stims, label: str, kind: str) -> dict:
        stims = as_stimuli(stims)
        t = self.engine_time()
        shifted = []
        for s in stims:
            stop = None if s.stop_ms is None else s.stop_ms + t
            shifted.append(s.between(s.start_ms + t, stop))
        n = int(np.unique(np.concatenate([s.targets for s in shifted])).size) if shifted else 0
        sid = self._next_id
        self._next_id += 1
        self.stimuli[sid] = {"label": label, "kind": kind, "stims": shifted, "n": n}
        targets = np.unique(np.concatenate([s.targets for s in shifted])) if shifted else np.zeros(0, np.int64)
        return {"id": sid, "label": label, "n": n, "targets": b64(targets.astype(np.uint32))}

    def n_driven(self) -> int:
        stims = self._all_stimuli()
        return int(np.unique(np.concatenate([s.targets for s in stims])).size) if stims else 0

    # -------------------------------------------------------- commands
    def command(self, c: dict) -> dict:
        cmd = c.get("cmd")
        with self.lock:
            fn = getattr(self, f"_cmd_{cmd}", None)
            if fn is None:
                return {"ok": False, "error": f"unknown command {cmd!r}"}
            try:
                out = fn(c) or {}
                out.setdefault("ok", True)
                return out
            except KeyError as exc:
                return {"ok": False, "error": str(exc).strip('"')}
            except Exception as exc:  # report to the UI
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _replace(self, c):
        if c.get("replace"):
            self.stimuli.clear()

    def _cmd_scenario(self, c):
        self._replace(c)
        sc = scenarios.get(c["name"])
        return self.add_stimulus(sc.build(self.brain), sc.title, "scenario:" + ",".join(sc.tags))

    def _cmd_smell(self, c):
        self._replace(c)
        conc = float(c.get("concentration", 1.0))
        st = olfaction.smell(self.brain, c.get("odor", "banana"), conc, left=c.get("left"), right=c.get("right"))
        return self.add_stimulus(st, f"smell {c.get('odor')} ({conc:g})", "smell")

    def _cmd_taste(self, c):
        self._replace(c)
        t = c.get("tastant", "sugar")
        st = taste_fn(self.brain, t, float(c.get("intensity", 1.0)), side=c.get("side", "both"))
        return self.add_stimulus(st, f"taste {t}", f"taste:{t}")

    def _cmd_vision(self, c):
        self._replace(c)
        kind = c.get("kind", "looming")
        b = self.brain
        az = float(c.get("azimuth", 45.0))
        if kind == "looming":
            st = vision.looming(b, azimuth=az, duration_ms=float(c.get("duration_ms", 400)))
        elif kind == "flash":
            st = vision.flash(b, stop_ms=80.0)
        elif kind == "spot":
            st = vision.spot(b, az, float(c.get("elevation", 0)), float(c.get("radius", 15)), color=c.get("color", "white"))
        elif kind == "grating":
            st = vision.grating(b, c.get("direction", "front_to_back"), stop_ms=float(c.get("duration_ms", 2000)))
        elif kind == "object":
            st = vision.moving_object(b, -80, 80, duration_ms=float(c.get("duration_ms", 1500)))
        else:
            raise KeyError(f"unknown vision stimulus {kind!r}")
        return self.add_stimulus(st, f"vision {kind}" + (f" @{az:+.0f}deg" if kind in ("looming", "spot") else ""), f"vision:{kind}")

    def _cmd_hear(self, c):
        self._replace(c)
        return self.add_stimulus(mechano.hear(self.brain, c.get("sound", "pulse_song")), f"hear {c.get('sound', 'pulse_song')}", "hearing")

    def _cmd_wind(self, c):
        self._replace(c)
        d = float(c.get("direction", 0))
        return self.add_stimulus(mechano.wind(self.brain, d), f"wind from {d:+.0f}deg", "wind")

    def _cmd_touch(self, c):
        self._replace(c)
        return self.add_stimulus(mechano.touch(self.brain, c.get("where", "eye")), f"touch {c.get('where', 'eye')}", "touch")

    def _cmd_temperature(self, c):
        self._replace(c)
        kind = c.get("kind", "hot")
        st = thermo.temperature(self.brain, kind, side=c.get("side", "both"))
        return self.add_stimulus(st, f"{kind} ({c.get('side', 'both')})", "temperature")

    def _cmd_stimulate(self, c):
        self._replace(c)
        ns = self.brain[c["name"]]
        if c.get("side") in ("left", "right"):
            ns = ns.where(side=c["side"])
        rate = float(c.get("rate", 150))
        return self.add_stimulus(Stimulus(ns.idx, rate, name=c["name"]), f"activate {c['name']} ({len(ns)}) @{rate:g}Hz", "neurons")

    def _cmd_remove(self, c):
        self.stimuli.pop(int(c["id"]), None)

    def _cmd_clear(self, c):
        self.stimuli.clear()

    def _cmd_silence(self, c):
        ns = self.brain[c["name"]]
        self.sim.silence(ns)
        if self.fly is not None:
            self.fly.sim.silence(ns)
        return {"n": len(ns), "silenced": b64(self.sim.silenced.astype(np.uint32))}

    def _cmd_unsilence(self, c):
        self.sim.unsilence()
        if self.fly is not None:
            self.fly.sim.unsilence()
        return {"silenced": ""}

    def _cmd_play(self, c):
        self.running = True

    def _cmd_pause(self, c):
        self.running = False

    def _cmd_speed(self, c):
        self.speed = float(np.clip(float(c.get("value", 1.0)), 0.05, 20.0))
        return {"speed": self.speed}

    def _cmd_reset(self, c):
        self.stimuli.clear()
        self.sim.reset()
        self.motor.reset()
        if self.mode == "fly":
            self._cmd_mode({"mode": "fly", "arena": self.arena_name})

    def _cmd_mode(self, c):
        mode = c.get("mode", "brain")
        if mode == "fly":
            from ..agent import Arena, VirtualFly

            self.arena_name = c.get("arena") or self.arena_name or "banana"
            fly_sim = Simulator(self.brain, seed=int(c.get("seed", 0)))
            if self.sim.silenced.size:
                fly_sim.silence(self.brain.at(self.sim.silenced))
            self.fly = VirtualFly(self.brain, Arena.preset(self.arena_name), x=float(c.get("x", 25)), y=float(c.get("y", 25)),
                                  heading_deg=float(c.get("heading", 45)), sim=fly_sim,
                                  turn_highpass_s=3.0 if self.brain.N > 20000 else None)
            self.mode = "fly"
            return {"arena": self.fly.arena.to_dict(), "field": self.odor_field()}
        self.mode = "brain"
        self.fly = None
        return {}

    def _cmd_fly_place(self, c):
        if self.fly is None:
            raise KeyError("not in fly mode")
        self.fly.x, self.fly.y = float(c["x"]), float(c["y"])
        if "heading" in c:
            self.fly.heading = np.radians(float(c["heading"]))

    def _cmd_arena_add(self, c):
        if self.fly is None:
            raise KeyError("not in fly mode")
        from ..agent import FoodSource, HeatZone, Predator

        kind = c.get("kind", "food")
        if kind == "food":
            item = FoodSource(float(c["x"]), float(c["y"]), c.get("odor", "banana"), c.get("taste", "sugar"),
                              name=c.get("name", c.get("odor", "food")))
        elif kind == "heat":
            item = HeatZone(float(c["x"]), float(c["y"]))
        elif kind == "predator":
            item = Predator(t_s=self.fly.t_ms / 1000 + 0.2, bearing_deg=float(c.get("bearing", 60)))
        else:
            raise KeyError(f"unknown item {kind!r}")
        self.fly.arena.items.append(item)
        self.fly._prepare()
        return {"arena": self.fly.arena.to_dict(), "field": self.odor_field()}

    def odor_field(self, n: int = 48) -> dict:
        a = self.fly.arena
        xs = np.linspace(0, a.width, n)
        ys = np.linspace(0, a.height, n)
        grid = np.zeros((n, n), dtype=np.float32)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                grid[j, i] = sum(a.odors_at(x, y).values())
        return {"n": n, "max": float(grid.max()), "data": b64(grid)}

    def _cmd_highlight(self, c):
        ns = self.brain[c["name"]]
        return {"name": c["name"], "n": len(ns), "indices": b64(ns.idx.astype(np.uint32))}

    def _cmd_color(self, c):
        return {"mode": c.get("mode", "class")}

    def _cmd_path(self, c):
        from ..analysis import strongest_path

        df = strongest_path(self.brain, c["source"], c["target"], min_fraction=float(c.get("min_fraction", 0.01)))
        return {"path": df.fillna(0).to_dict("records")}

    def _cmd_influence(self, c):
        from ..analysis import influence

        v = influence(self.brain, c["source"], n_hops=int(c.get("hops", 3))).fillna(0).to_numpy(np.float32)
        m = float(np.abs(v).max()) or 1.0
        return {"values": b64((v / m).astype(np.float32)), "max": m}

    def _cmd_challenge(self, c):
        kinds = [e["kind"] for e in self.stimuli.values()]
        drive = self.fly.last_drive if (self.mode == "fly" and self.fly is not None and hasattr(self.fly, "last_drive")) else self.motor.drive
        return lessons.check_challenge(c["id"], {"drive": drive, "kinds": kinds, "n_driven": self.n_driven()})

    def _cmd_neuron(self, c):
        i = int(c["index"])
        d = self.brain.describe(self.brain.at([i]))
        clean = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating, float)):
                v = None if np.isnan(v) else round(float(v), 3)
            elif isinstance(v, list):
                v = [{kk: (int(vv) if isinstance(vv, np.integer) else (round(float(vv), 4) if isinstance(vv, (np.floating, float)) else vv))
                      for kk, vv in row.items()} for row in v]
            clean[k] = v
        return {"neuron": clean}

    def _cmd_search(self, c):
        df = self.brain.search(c.get("q", ""), limit=int(c.get("limit", 20)))
        return {"results": df.to_dict("records")}

    # ------------------------------------------------------------ jobs
    def _cmd_job(self, c):
        kind = c.get("kind")
        jid = self._job_id
        self._job_id += 1
        job = {"id": jid, "kind": kind, "status": "running", "progress": "starting", "result": None}
        self.jobs[jid] = job
        threading.Thread(target=self._run_job, args=(job, c), daemon=True).start()
        return {"job": jid}

    def _run_job(self, job, c):
        try:
            if job["kind"] == "conditioning":
                from ..learning import OlfactoryConditioning

                job["progress"] = "pre-test, training and post-test (this takes a while on big brains)..."
                exp = OlfactoryConditioning(self.brain, cs_plus=c.get("cs_plus", "banana"), cs_minus=c.get("cs_minus", "acetic_acid"),
                                            reinforcer=c.get("reinforcer", "reward"))
                res = exp.run(n_trials=int(c.get("trials", 2)), odor_ms=float(c.get("odor_ms", 600)), test_ms=float(c.get("test_ms", 400)))
                w = res.weights.sort_values("weight_ratio").head(10)
                job["result"] = {"summary": res.summary(), "valence": res.valence.round(3).to_dict(),
                                 "learning_index": round(res.learning_index, 3),
                                 "curve": res.curve.round(3).to_dict("records"),
                                 "weights": w[["cell_type", "side", "nt", "weight_ratio"]].round(3).to_dict("records")}
            elif job["kind"] == "screen":
                from ..learning import activation_screen

                job["progress"] = "activating candidate cell types one by one (rate model)..."
                df = activation_screen(self.brain, c.get("target", "MN9"), top=int(c.get("top", 12)))
                job["result"] = {"table": df.round(2).to_dict("records")}
            else:
                raise KeyError(f"unknown job {job['kind']!r}")
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "error"
            job["result"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------ info
    def info(self) -> dict:
        b = self.brain
        from ..agent import Arena

        return {
            "name": b.name,
            "title": b.meta.get("title", b.name),
            "description": b.meta.get("description", ""),
            "citation": b.meta.get("citation", ""),
            "license": b.meta.get("license", ""),
            "layout": b.meta.get("layout", "anatomical"),
            "transmitter_policy": b.meta.get("transmitter_policy", "shiu2024"),
            "N": b.N,
            "n_connections": b.n_connections,
            "n_synapses": b.n_synapses,
            "classes": [{"name": n or "unannotated", "color": c, "count": int((self.class_codes == k).sum())}
                        for k, (n, c) in enumerate(zip(self.class_names, self.class_colors))],
            "nts": [{"name": n or "unknown", "color": c, "count": int((self.nt_codes == k).sum())}
                    for k, (n, c) in enumerate(zip(self.nt_names, self.nt_colors))],
            "sides": [{"name": n or "unknown", "color": c} for n, c in zip(self.side_names, self.side_colors)],
            "scenarios": [{"name": s.name, "title": s.title, "story": s.story, "real_fly": s.real_fly, "tags": list(s.tags),
                           "watch": list(s.watch)} for s in scenarios.SCENARIOS.values()],
            "odors": [{"name": o.name, "valence": o.valence, "description": o.description} for o in olfaction.ODORS.values()],
            "tastants": sorted(__import__("openfly.senses.taste", fromlist=["TASTANTS"]).TASTANTS),
            "arenas": list(Arena.PRESETS),
            "channels": [{"name": ch.name, "description": ch.description, "mode": ch.mode} for ch in motor.CHANNELS.values()
                         if ch.name in self.motor.channels],
            "lessons": lessons.LESSONS,
            "challenges": lessons.CHALLENGES,
            "chunk_ms": self.chunk_ms,
        }

    def geometry(self) -> bytes:
        """Float32 positions (N x 3, normalised, y up) followed by uint8 class/nt/side codes."""
        P = self.brain.positions.astype(np.float64)
        c = np.nanmedian(P, axis=0)
        P = P - c
        s = np.nanmax(np.abs(P)) or 1.0
        P = (P / s).astype(np.float32)
        P[:, 1] *= -1  # FlyWire y grows ventrally; show dorsal up
        P = np.nan_to_num(P)
        codes = np.stack([self.class_codes, self.nt_codes, self.side_codes], axis=1).astype(np.uint8)
        return P.tobytes() + codes.tobytes()
