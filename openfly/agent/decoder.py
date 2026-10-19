"""Population decoders: read intentions from many neurons, as a brain-computer interface does.

The default steering readout uses one cell type (DNa02, left vs right).
On the real FlyWire brain that single readout is weak and biased under
smell, but *many* descending neurons respond differently to an odour on
the left vs the right. A decoder calibrated on those responses recovers
the side. This is the same idea as decoding a reach direction from motor
cortex in a BCI.

    dec = SteeringDecoder.fit(brain, odor="banana")
    fly = VirtualFly(brain, "banana", steering=dec)

On FlyWire (curated transmitters) a decoder reading the ~170 responsive
neurons of the ``steering_premotor`` set (LAL/SPS/IPS, incl. PFL3) decoded
held-out conditions correctly: odour stronger on the left -0.19, right
+0.21, symmetric -0.06. Reading descending neurons alone was less reliable.
In closed loop it steered about as well as the high-passed DNa02 readout,
not better (docs/09-virtual-fly.md): it was calibrated on steady odours,
and a walking fly's odour input changes all the time.

**Calibration session.** ``fit`` simulates the odour at several
intensities, stronger on the left (target -1), stronger on the right
(+1) and symmetric (0), and fits a ridge regression from the firing
rates of candidate neurons (default: the ``steering_premotor`` set,
falling back to all descending neurons) to the target. Symmetric conditions at different intensities teach the decoder
to ignore overall level. Held-out conditions report how well it generalises.
The decoder only *reads* the brain and changes nothing in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..senses import olfaction


@dataclass
class SteeringDecoder:
    idx: np.ndarray       # neurons read (model indices)
    weights: np.ndarray   # ridge weights (per neuron)
    bias: float
    info: dict = field(default_factory=dict)

    def __call__(self, rates: np.ndarray) -> float:
        """Turn signal in [-1, 1] (> 0 = right) from the rates (Hz) of ``self.idx``."""
        return float(np.clip(self.weights @ rates + self.bias, -1.0, 1.0))

    def save(self, path) -> str:
        np.savez(path, idx=self.idx, weights=self.weights, bias=self.bias, info=np.array([repr(self.info)]))
        return str(path)

    @classmethod
    def load(cls, path) -> "SteeringDecoder":
        import ast

        z = np.load(path, allow_pickle=False)
        try:
            info = ast.literal_eval(str(z["info"][0]))
        except (ValueError, SyntaxError):
            info = {}
        return cls(z["idx"], z["weights"], float(z["bias"]), info)

    @classmethod
    def fit(cls, brain, odor: str = "banana", intensities=(0.4, 0.7, 1.0), ratio: float = 0.25,
            duration_ms: float = 1000.0, candidates="steering_premotor", n_trials: int = 4, ridge: float = 200.0,
            seed: int = 0) -> "SteeringDecoder":
        """Calibrate on simulated odour presentations (see module docstring).

        Takes ~6 minutes on FlyWire with numba (9 conditions x ``n_trials`` x
        ``duration_ms``, plus 3 held-out checks); save the result with :meth:`save`.
        """
        from ..sim.simulator import Simulator

        try:
            cand = brain[candidates].idx
        except KeyError:
            cand = np.zeros(0, np.int64)
        if cand.size == 0:
            cand = brain["descending"].idx
        X, y = [], []

        def run(left, right, k):
            sim = Simulator(brain, seed=seed + k)
            sim.add(olfaction.smell(brain, odor, left=left, right=right))
            return sim.run(duration_ms, n_trials=n_trials).rates_by_trial()[:, cand]

        k = 0
        for c in intensities:
            for (l, r), target in (((c, c * ratio), -1.0), ((c * ratio, c), 1.0), ((c, c), 0.0)):
                R = run(l, r, k)
                k += 1
                X.append(R)
                y.extend([target] * R.shape[0])
        X = np.vstack(X)
        y = np.asarray(y)
        used = X.max(axis=0) > 0
        if not used.any():
            raise ValueError(f"no {candidates} neuron responds to {odor}")
        Xu = X[:, used]
        mu = Xu.mean(axis=0)
        Xc = Xu - mu
        A = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
        w = np.linalg.solve(A, Xc.T @ (y - y.mean()))
        b = float(y.mean() - mu @ w)
        idx = cand[used]
        pred = np.clip(Xu @ w + b, -1, 1)
        # held-out check: intensities and ratios not used for fitting
        test = []
        for j, (l, r, target) in enumerate(((0.85, 0.85 * 0.6, -1), (0.85 * 0.6, 0.85, 1), (0.55, 0.55, 0))):
            Rt = run(l, r, 100 + j).mean(axis=0)[used]
            test.append((target, float(np.clip(Rt @ w + b, -1, 1))))
        order = np.argsort(-np.abs(w))[:8]
        info = {
            "odor": odor, "n_neurons": int(used.sum()), "candidates": candidates if isinstance(candidates, str) else "custom",
            "train_accuracy": float(np.mean(np.sign(np.round(pred, 1)) == np.sign(y))),
            "held_out": test,
            "top": brain.neurons.iloc[idx[order]][["cell_type", "side"]].values.tolist(),
        }
        return cls(idx, w, b, info)
