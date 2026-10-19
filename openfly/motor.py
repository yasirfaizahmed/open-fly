"""From spikes to behaviour: reading the brain's motor output.

The brain talks to the body through ~1,300 descending neurons and a few
motor neurons in the head. A handful of them are well enough understood
to read as "the fly wants to do X". Each :class:`MotorChannel` names one
such group and how to turn its firing into a behavioural drive between
0 and 1 (or -1 to 1 for turning).

Default channels:

==============  ================  ====================================================
channel         neurons           meaning
==============  ================  ====================================================
feeding         MN9 (CB0701)      proboscis extension (Shiu et al. 2024)
walk_forward    DNp09 ("P9")      forward walking (Bidaye et al. 2020)  [likely]
walk_backward   MDN               backward walking (Bidaye et al. 2014)
turn            DNa02 left/right  steering; >0 = turn right (Yang et al. 2024)
escape          giant fiber       take-off jump (von Reyn et al. 2014)
receptivity     pC1               female receptivity / social state (Zhou et al. 2014)  [likely]
grooming        groom_DN          toy brain only (no clean FlyWire marker yet)
==============  ================  ====================================================

These are deliberate simplifications. Real behaviour is a population
code across many DNs (Braun et al. 2024). Add channels with
:func:`add_channel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorChannel:
    name: str
    target: str
    description: str
    reference: str = ""
    mode: str = "mean"  # "mean" -> 0..1, "lr" -> -1..1 (right minus left)
    half_rate_hz: float = 20.0  # rate that counts as half-maximal drive
    confidence: str = "established"


CHANNELS: dict[str, MotorChannel] = {}


def add_channel(channel: MotorChannel) -> MotorChannel:
    CHANNELS[channel.name] = channel
    return channel


for _c in [
    MotorChannel("feeding", "MN9", "proboscis extension / feeding", "Shiu et al. 2024 Nature", half_rate_hz=20.0),
    MotorChannel("walk_forward", "DNp09", "forward walking", "Bidaye et al. 2020 Neuron", confidence="likely"),
    MotorChannel("walk_backward", "MDN", "backward walking (moonwalking)", "Bidaye et al. 2014 Science"),
    MotorChannel("turn", "DNa02", "steering (>0 right, <0 left)", "Yang et al. 2024 Cell", mode="lr"),
    MotorChannel("escape", "giant_fiber", "escape take-off", "von Reyn et al. 2014 Nat Neurosci", half_rate_hz=10.0),
    MotorChannel("receptivity", "pC1", "female receptivity / social arousal (pC1)", "Zhou et al. 2014 Neuron",
                 confidence="likely"),
    MotorChannel("grooming", "groom_DN", "head/eye grooming (toy brain)", "Hampel et al. 2015 eLife", confidence="toy"),
]:
    add_channel(_c)


def _sets(brain, channels):
    cache = brain.__dict__.setdefault("_motor_cache", {})
    out = {}
    for ch in channels:
        if ch.name not in cache:
            try:
                ns = brain[ch.target]
            except KeyError:
                ns = None
            cache[ch.name] = ns if ns is not None and len(ns) else None
        out[ch.name] = cache[ch.name]
    return out


def read_behavior(brain, rates: np.ndarray, channels=None) -> dict:
    """Behavioural drives from per-neuron firing rates (Hz).

    Returns ``{channel: {"drive", "rate_hz", "left_hz", "right_hz", "neurons"}}``;
    channels whose neurons don't exist in this brain are left out.
    """
    chans = [CHANNELS[c] if isinstance(c, str) else c for c in (channels or CHANNELS.values())]
    sets = _sets(brain, chans)
    out = {}
    for ch in chans:
        ns = sets[ch.name]
        if ns is None:
            continue
        r = rates[ns.idx]
        side = brain.neurons["side"].to_numpy()[ns.idx]
        left = float(r[side == "left"].mean()) if (side == "left").any() else 0.0
        right = float(r[side == "right"].mean()) if (side == "right").any() else 0.0
        mean = float(r.mean())
        if ch.mode == "lr":
            drive = (right - left) / (right + left + ch.half_rate_hz)
        else:
            drive = mean / (mean + ch.half_rate_hz)
        out[ch.name] = {
            "drive": round(float(drive), 4),
            "rate_hz": round(mean, 2),
            "left_hz": round(left, 2),
            "right_hz": round(right, 2),
            "neurons": len(ns),
            "description": ch.description,
        }
    return out


def describe(behavior: dict) -> str:
    """One line per channel, e.g. ``feeding   ########..  0.82  (MN9 91 Hz)``."""
    lines = []
    for name, b in behavior.items():
        d = b["drive"]
        if CHANNELS.get(name) and CHANNELS[name].mode == "lr":
            arrow = "right" if d > 0.05 else "left" if d < -0.05 else "straight"
            bar = "<" * int(round(-d * 10)) if d < 0 else ">" * int(round(d * 10))
            lines.append(f"{name:<14}{bar:<10} {d:+.2f}  {arrow} (L {b['left_hz']:.0f} / R {b['right_hz']:.0f} Hz)")
        else:
            bar = "#" * int(round(d * 10))
            lines.append(f"{name:<14}{bar:<10} {d:.2f}  ({b['rate_hz']:.0f} Hz)")
    return "\n".join(lines)


class MotorMonitor:
    """Live behaviour readout from a stream of spikes (exponentially smoothed).

    Feed it the spikes of each simulation chunk with :meth:`update`; it
    returns the same drives as :func:`read_behavior`, smoothed with time
    constant ``tau_ms``. Used by the playground and the virtual fly.
    """

    def __init__(self, brain, channels=None, tau_ms: float = 100.0):
        chans = [CHANNELS[c] if isinstance(c, str) else c for c in (channels or CHANNELS.values())]
        sets = _sets(brain, chans)
        side = brain.neurons["side"].to_numpy()
        self.brain = brain
        self.tau_ms = tau_ms
        self.channels = {}
        for ch in chans:
            ns = sets[ch.name]
            if ns is not None:
                s = side[ns.idx]
                self.channels[ch.name] = (ns.idx, s == "left", s == "right", ch)
        self.ema: dict[str, float] = {}
        self.drive: dict[str, float] = {}

    def update(self, neurons: np.ndarray, window_ms: float) -> dict:
        counts = np.bincount(neurons, minlength=self.brain.N) if neurons.size else None
        win_s = window_ms / 1000.0
        alpha = min(1.0, window_ms / self.tau_ms)
        out = {}
        for name, (idx, is_l, is_r, ch) in self.channels.items():
            if counts is None:
                l = r = m = 0.0
            else:
                c = counts[idx] / win_s
                l = float(c[is_l].mean()) if is_l.any() else 0.0
                r = float(c[is_r].mean()) if is_r.any() else 0.0
                m = float(c.mean())
            for key, val in ((f"{name}_l", l), (f"{name}_r", r), (name, m)):
                self.ema[key] = (1 - alpha) * self.ema.get(key, 0.0) + alpha * val
            L, R, M = self.ema[f"{name}_l"], self.ema[f"{name}_r"], self.ema[name]
            if ch.mode == "lr":
                out[name] = (R - L) / (R + L + ch.half_rate_hz)
            else:
                out[name] = M / (M + ch.half_rate_hz)
        self.drive = out
        return out

    def rates(self) -> dict:
        """Smoothed rates (Hz): ``{channel: mean}`` plus ``channel_l`` / ``channel_r``."""
        return dict(self.ema)

    def reset(self) -> None:
        self.ema.clear()
        self.drive.clear()
