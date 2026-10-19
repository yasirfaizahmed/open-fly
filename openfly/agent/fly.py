"""The virtual fly: a connectome brain in a body, in a world, in closed loop.

Every body step (default 20 ms):

1. **sense**: the fly's antennae sample odour concentrations (left and
   right separately), its labellum tastes whatever it stands on, its eyes
   see food items and incoming predators at their bearing, and its arista
   feels heat. All of that becomes stimuli for the brain.
2. **think**: the whole-brain LIF simulation advances by the same 20 ms.
3. **act**: descending/motor neuron activity is read out (see
   :mod:`openfly.motor`) and moves the body: forward/backward walking,
   turning, feeding, escape jumps.

The body itself is deliberately simple: a point with a heading, like
the "fly on a ball" of a treadmill rig. Detailed biomechanics are what
NeuroMechFly (Lobato-Rios et al. 2022; Wang-Chen et al. 2024) is for;
this agent's motor output could drive it.

Navigation uses two strategies real flies use:

* **bilateral comparison** (tropotaxis): each antenna has its own ORNs, so
  a stronger smell on the left drives the left-side circuits harder and
  the steering readout turns the fly left.
* **temporal comparison** (klinokinesis): the agent watches its own
  brain's lateral horn (the innate olfactory centre). While activity there
  is rising it walks straighter; when it falls it turns more. The
  activity is read from the simulated neurons, and the run/turn policy is
  a simple body-level rule (``klinokinesis=0`` turns it off).

ORNs adapt (Weber's law): each antenna responds to the *log ratio* of the
current concentration to its recent average (``orn_gain`` sets the
steepness), so small left/right or moment-to-moment differences stay
visible at any absolute concentration.

Two honest compromises, both switchable:

* ``explore=True`` adds a small baseline walk and random turning when the
  brain is quiet. Real flies have central pattern generators and internal
  drive that the brain-only model lacks. ``explore=False`` is purely brain-driven.
* ``turn_highpass_s`` (off by default) removes slow, constant left/right
  biases of the steering readout, so only *changes* in asymmetry steer.
  Useful for FlyWire, whose DNa02 readout has an intrinsic left bias
  under olfactory input (try ``turn_highpass_s=3``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import motor as motor_mod
from ..senses import olfaction, vision
from ..senses.taste import taste as taste_fn
from ..senses.base import Stimulus
from ..sim.simulator import Simulator
from .arena import Arena


class VirtualFly:
    def __init__(self, brain, arena: Arena | str = "banana", x: float = 25.0, y: float = 25.0,
                 heading_deg: float = 45.0, step_ms: float = 20.0, seed: int = 0, explore: bool = True,
                 turn_highpass_s: float | None = None, antenna_offset_mm: float = 2.0, max_speed: float = 20.0,
                 base_speed: float = 6.0, max_turn_deg_s: float = 500.0, sim: Simulator | None = None,
                 klinokinesis: float = 3.0, klinokinesis_target: str = "LHN", adaptation_s: float = 1.5,
                 orn_gain: float = 3.0):
        self.brain = brain
        self.arena = Arena.preset(arena) if isinstance(arena, str) else arena
        self.x, self.y, self.heading = float(x), float(y), math.radians(heading_deg)
        self.step_ms = step_ms
        self.rng = np.random.default_rng(seed)
        self.explore = explore
        self.turn_hp = turn_highpass_s
        self.antenna = antenna_offset_mm
        self.max_speed, self.base_speed, self.max_turn = max_speed, base_speed, math.radians(max_turn_deg_s)
        self.sim = sim or Simulator(brain, seed=seed)
        self.t_ms = 0.0
        self.energy = 0.0
        self.alive_jumps = 0
        self.state = "exploring"
        self.trail: list[dict] = []
        self.events: list[dict] = []
        #: extra stimuli applied on top of what the fly senses (e.g. optogenetics from the playground)
        self.extra_stimuli: list = []
        self._turn_baseline = 0.0
        self._jump_cooldown = 0.0
        self.klinokinesis = klinokinesis
        self._kk_target = klinokinesis_target
        self.adaptation_s = adaptation_s
        self.orn_gain = orn_gain
        self._adapt: dict[str, float] = {}
        self._kk_fast = self._kk_slow = 0.0
        self._kk_change = 0.0
        self._prepare()

    # ------------------------------------------------------------ set-up
    def _prepare(self) -> None:
        b = self.brain
        side = b.neurons["side"].to_numpy()
        self._odors = {}
        for it in self.arena.items:
            if getattr(it, "odor", None) and it.odor not in self._odors:
                st = olfaction.smell(b, it.odor, 1.0)
                w = st.rates / olfaction._saturate(1.0) / 150.0 if len(st) else st.rates
                self._odors[it.odor] = (st.targets, w, side[st.targets])
        self._tastes = {}
        for it in self.arena.items:
            if getattr(it, "taste", None) and it.taste not in self._tastes:
                self._tastes[it.taste] = taste_fn(b, it.taste, 1.0)
        self._vf = vision.VisualField.of(b)
        self._heat = {}
        for s in ("left", "right"):
            try:
                self._heat[s] = b["hot_TRN"].where(side=s).idx
            except KeyError:
                self._heat[s] = np.zeros(0, np.int64)
        self.motor = motor_mod.MotorMonitor(b, ["feeding", "walk_forward", "walk_backward", "turn", "escape"])
        self._looming = None
        try:
            self._kk_idx = b[self._kk_target].idx
        except KeyError:
            self._kk_idx = np.zeros(0, np.int64)

    # ------------------------------------------------------------- senses
    def _antennae(self):
        h = self.heading
        fx, fy = self.x + 0.8 * math.cos(h), self.y + 0.8 * math.sin(h)
        lx, ly = fx + self.antenna * math.cos(h + math.pi / 2), fy + self.antenna * math.sin(h + math.pi / 2)
        rx, ry = fx + self.antenna * math.cos(h - math.pi / 2), fy + self.antenna * math.sin(h - math.pi / 2)
        return (lx, ly), (rx, ry)

    def sense(self) -> list[Stimulus]:
        stims: list[Stimulus] = []
        (lx, ly), (rx, ry) = self._antennae()
        cl, cr = self.arena.odors_at(lx, ly), self.arena.odors_at(rx, ry)
        self.last_sense = {"odor_left": cl, "odor_right": cr}
        dt_s = self.step_ms / 1000.0
        for odor, (idx, w, s) in self._odors.items():
            c_l, c_r = cl.get(odor, 0.0), cr.get(odor, 0.0)
            mean = (c_l + c_r) / 2
            a = self._adapt.get(odor, mean)
            if self.adaptation_s:
                a += (mean - a) * min(1.0, dt_s / self.adaptation_s)
            self._adapt[odor] = a
            if max(c_l, c_r) < 0.01:
                continue

            def resp(c, a=a):
                # Weber-law ORNs: half-maximal at the adapted level, responding to log-ratio changes
                if not self.adaptation_s:
                    return olfaction._saturate(c)
                return float(np.clip(0.5 + self.orn_gain * np.log((c + 0.01) / (a + 0.01)), 0.0, 1.0))

            act = np.where(s == "left", resp(c_l), np.where(s == "right", resp(c_r), resp(mean)))
            stims.append(Stimulus(idx, 150.0 * w * act, name=f"smell:{odor}"))
        food = self.arena.food_at(self.x, self.y, reach_mm=1.0)
        self.last_sense["touching"] = food.name if food else None
        if food is not None and food.taste in self._tastes:
            stims.append(self._tastes[food.taste])
        # vision: food items as bright spots at their bearing
        for it in self.arena.visible_objects():
            dx, dy = it.x - self.x, it.y - self.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            bearing = math.degrees(math.atan2(dy, dx) - self.heading)
            bearing = (bearing + 180) % 360 - 180
            az = -bearing  # azimuth positive = fly's right; bearing positive = counter-clockwise = left
            size = math.degrees(2 * math.atan(it.radius_mm / dist))
            if abs(az) < 165 and size > 1.0:
                # A static object is not a threat: its image is capped at 20 deg and driven
                # gently, so approaching food doesn't look like a looming predator.
                idx, d = self._vf.select(az, 0.0, min(max(size / 2, 4.0), 10.0), vision.COLOR_CHANNELS["yellow"])
                if idx.size:
                    stims.append(Stimulus(idx, min(70.0, 30.0 + 3.0 * size), name=f"see:{it.name}"))
        # heat on each side
        h_l, h_r = self.arena.heat_at(lx, ly), self.arena.heat_at(rx, ry)
        for s, hv in (("left", h_l), ("right", h_r)):
            if hv > 0.05 and self._heat[s].size:
                stims.append(Stimulus(self._heat[s], 150.0 * min(hv, 1.0), name=f"heat:{s}"))
        # predators: a looming stimulus when its time comes
        for p in self.arena.predators():
            if not p.fired and self.t_ms >= p.t_s * 1000:
                p.fired = True
                self._looming = vision.looming(self.brain, azimuth=p.bearing_deg, start_ms=self.sim.engine.time_ms,
                                               duration_ms=p.duration_ms)
                self.events.append({"t_s": self.t_ms / 1000, "event": f"predator attacks from {p.bearing_deg:+.0f} deg"})
        if self._looming is not None:
            stims.extend(self._looming)
            if self.sim.engine.time_ms > self._looming[0].stop_ms:
                self._looming = None
        self.last_sense["heat"] = (h_l, h_r)
        return stims

    # -------------------------------------------------------------- motor
    def _read(self, neurons: np.ndarray) -> dict:
        return self.motor.update(neurons, self.step_ms)

    def act(self, drive: dict) -> None:
        dt = self.step_ms / 1000.0
        feeding = drive.get("feeding", 0.0)
        food = self.arena.food_at(self.x, self.y, reach_mm=1.0)
        turn_raw = drive.get("turn", 0.0)
        if self.turn_hp:
            a = dt / self.turn_hp
            self._turn_baseline = (1 - a) * self._turn_baseline + a * turn_raw
            turn = turn_raw - self._turn_baseline
        else:
            turn = turn_raw
        self._jump_cooldown = max(0.0, self._jump_cooldown - dt)
        if drive.get("escape", 0.0) > 0.5 and self._jump_cooldown == 0.0:
            jump = 15.0 + 10.0 * self.rng.random()
            self.heading += self.rng.normal(0, 0.8)
            self._move(jump)
            self._jump_cooldown = 1.0
            self.alive_jumps += 1
            self.state = "escaping"
            self.events.append({"t_s": self.t_ms / 1000, "event": "escape jump (giant fiber)"})
            return
        if food is not None and feeding > 0.25:
            self.state = "feeding"
            self.energy += feeding * dt
            if not self.events or self.events[-1]["event"] != f"feeding on {food.name}":
                self.events.append({"t_s": self.t_ms / 1000, "event": f"feeding on {food.name}"})
            return
        fwd = drive.get("walk_forward", 0.0)
        back = drive.get("walk_backward", 0.0)
        speed = self.max_speed * fwd - 0.6 * self.max_speed * back
        if self.explore:
            speed += self.base_speed * (1 - back)
        if food is not None:
            speed *= 0.1  # leg taste receptors stop the fly on food, so the brain can taste it
        omega = -self.max_turn * turn  # turn > 0 means right = clockwise = negative angle
        speed /= 1.0 + 4.0 * abs(turn)  # flies slow down while turning sharply
        if self.explore:
            # klinokinesis: straighter while the lateral horn's activity rises, more turning as it falls
            noise = 1.2 * float(np.clip(1.0 - self.klinokinesis * self._kk_change, 0.25, 3.0))
            omega += self.rng.normal(0, noise)
            speed *= float(np.clip(1.0 + 0.5 * self._kk_change, 0.6, 1.5))
        self.heading += omega * dt
        self._move(speed * dt)
        self.state = "backing up" if back > max(fwd, 0.2) else "walking" if abs(speed) > 0.5 else "still"

    def _move(self, d: float) -> None:
        nx = self.x + d * math.cos(self.heading)
        ny = self.y + d * math.sin(self.heading)
        W, H = self.arena.width, self.arena.height
        if nx < 0 or nx > W:
            self.heading = math.pi - self.heading
            nx = min(max(nx, 0.0), W)
        if ny < 0 or ny > H:
            self.heading = -self.heading
            ny = min(max(ny, 0.0), H)
        self.x, self.y = nx, ny

    # --------------------------------------------------------------- loop
    def step(self) -> dict:
        """One sense-think-act cycle. Returns the fly's state (also appended to ``trail``)."""
        self.sim.stimuli = self.sense() + list(self.extra_stimuli)
        times, neurons = self.sim.advance(self.step_ms)
        drive = self._read(neurons)
        if self._kk_idx.size:
            n_kk = np.isin(neurons, self._kk_idx).sum() / (self._kk_idx.size * self.step_ms / 1000.0)
            dt_s = self.step_ms / 1000.0
            self._kk_fast += (n_kk - self._kk_fast) * min(1.0, dt_s / 0.2)
            self._kk_slow += (n_kk - self._kk_slow) * min(1.0, dt_s / 1.5)
            self._kk_change = (self._kk_fast - self._kk_slow) / (self._kk_slow + 2.0)
        self.act(drive)
        self.t_ms += self.step_ms
        rec = {"t_s": round(self.t_ms / 1000, 3), "x": self.x, "y": self.y, "heading_deg": math.degrees(self.heading) % 360,
               "state": self.state, "energy": self.energy, "spikes": int(neurons.size),
               **{f"drive_{k}": round(v, 3) for k, v in drive.items()}}
        self.trail.append(rec)
        self.last_spikes = neurons
        self.last_drive = drive
        return rec

    def run(self, seconds: float = 10.0, progress: bool = False) -> pd.DataFrame:
        n = int(round(seconds * 1000 / self.step_ms))
        for k in range(n):
            self.step()
            if progress and (k + 1) % max(1, n // 10) == 0:
                r = self.trail[-1]
                print(f"  t={r['t_s']:5.1f}s  pos=({r['x']:5.1f},{r['y']:5.1f})  {r['state']:<10} energy={r['energy']:.2f}", flush=True)
        return self.trajectory()

    def trajectory(self) -> pd.DataFrame:
        return pd.DataFrame(self.trail)

    def distance_to(self, name: str | None = None) -> float:
        foods = [it for it in self.arena.visible_objects() if name is None or it.name == name]
        return min(math.hypot(it.x - self.x, it.y - self.y) for it in foods) if foods else float("nan")

    def plot(self, ax=None):
        from ..viz.plots import plot_trajectory

        return plot_trajectory(self, ax=ax)
