"""Teaching the fly: dopamine-gated plasticity in the mushroom body.

How a real fly learns that an odour means food (or danger):

1. An odour activates a small, odour-specific subset of the ~5,000
   **Kenyon cells** (KCs), about 5%.
2. KCs synapse onto ~35 types of **mushroom body output neurons**
   (MBONs). Some MBONs push the fly to approach, others to avoid
   (Aso et al. 2014).
3. **Dopaminergic neurons** (DANs) carry teaching signals: PAM DANs fire
   for reward (sugar), PPL1 DANs for punishment (shock, bitter). Each DAN
   type innervates one *compartment*, so it reaches specific MBONs.
4. When KCs are active *while* dopamine arrives, their synapses onto that
   compartment's MBON get weaker (long-term depression; Hige et al. 2015).
   Reward weakens KC -> avoidance-MBON synapses, so the odour now drives
   the fly toward it.

:class:`MushroomBodyPlasticity` implements step 4 as a three-factor rule
on the real connectome. Compartments are not hand-coded: each MBON listens
to the DANs that actually synapse onto it (DAN -> MBON connections in the
data reproduce the textbook pairs, e.g. PPL1-gamma1 -> MBON-gamma1pedc).

    dw/dt = -lr * kc_trace[pre] * dopamine[post] * w0     (bounded to [w_min, w0])

:class:`OlfactoryConditioning` runs a full differential-conditioning
experiment (pre-test, training, post-test), the computational version of
the classic T-maze assay (Tully & Quinn 1985).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: MBON valence by transmitter (rule of thumb from Aso et al. 2014 eLife 3:e04580):
#: glutamatergic MBONs promote avoidance, cholinergic and GABAergic ones approach.
VALENCE_BY_NT = {"glutamate": -1.0, "acetylcholine": 1.0, "gaba": 1.0}


class MushroomBodyPlasticity:
    """Attach dopamine-gated KC->MBON depression to a :class:`~openfly.Simulator`.

    Parameters
    ----------
    lr:
        learning rate (fraction of the original weight lost per ms at unit
        KC trace and unit dopamine).
    tau_kc_ms, tau_da_ms:
        decay of the KC eligibility trace and of the dopamine signal.
    w_min:
        floor as a fraction of the original weight (synapses never vanish).
    min_dan_synapses:
        DAN -> MBON connections weaker than this are ignored when building
        compartments.
    da_threshold:
        dopamine (trace units, ~spikes within tau) below which nothing is
        learned. In the whole-brain model odours alone excite some DANs a
        little; the threshold keeps that from writing memories.
    """

    def __init__(self, sim, kc="KC", mbon="MBON", dan="DAN", lr: float = 0.002, tau_kc_ms: float = 60.0,
                 tau_da_ms: float = 60.0, w_min: float = 0.05, min_dan_synapses: int = 1, update_every_steps: int = 10,
                 da_threshold: float = 1.0):
        self.sim = sim
        brain = sim.brain
        self.kc, self.mbon, self.dan = brain[kc], brain[mbon], brain[dan]
        eng = sim.engine
        N = brain.N
        self.kc_local = np.full(N, -1, np.int64)
        self.kc_local[self.kc.idx] = np.arange(len(self.kc))
        self.dan_local = np.full(N, -1, np.int64)
        self.dan_local[self.dan.idx] = np.arange(len(self.dan))
        mbon_local = np.full(N, -1, np.int64)
        mbon_local[self.mbon.idx] = np.arange(len(self.mbon))

        # CSR positions of every KC -> MBON synapse in the engine
        starts = eng._indptr[self.kc.idx]
        lens = eng._indptr[self.kc.idx + 1] - starts
        pos = np.concatenate([np.arange(s, s + l) for s, l in zip(starts, lens)]) if lens.sum() else np.zeros(0, np.int64)
        pre_local = np.repeat(np.arange(len(self.kc)), lens)
        post = eng._indices[pos]
        keep = mbon_local[post] >= 0
        self.pos = pos[keep]
        self.pre = pre_local[keep]
        self.post = mbon_local[post[keep]]
        self.w0 = eng.syn_weight[self.pos].copy()

        # compartments from the wiring: an MBON's dopamine is the synapse-weighted
        # mean activity of the DANs that contact it (strong partners dominate)
        C = abs(brain.W[self.dan.idx][:, self.mbon.idx]).toarray().astype(np.float64)
        C[C < min_dan_synapses] = 0.0
        total = C.sum(axis=0)
        self.compartment = C / np.maximum(total, 1.0)
        self.plastic_mbons = total > 0

        dt = sim.params.dt_ms
        self.decay_kc = np.exp(-dt / tau_kc_ms)
        self.decay_da = np.exp(-dt / tau_da_ms)
        self.lr, self.w_min, self.da_threshold = lr, w_min, da_threshold
        self.every = int(update_every_steps)
        self.kc_trace = np.zeros(len(self.kc))
        self.da_trace = np.zeros(len(self.dan))
        self.enabled = True
        self._k = 0
        self.history: list[dict] = []
        sim.on_step(self._on_step)

    # ------------------------------------------------------------ rule
    def _on_step(self, engine, spikes):
        self.kc_trace *= self.decay_kc
        self.da_trace *= self.decay_da
        if spikes.size:
            k = self.kc_local[spikes]
            k = k[k >= 0]
            if k.size:
                np.add.at(self.kc_trace, k, 1.0)
            d = self.dan_local[spikes]
            d = d[d >= 0]
            if d.size:
                np.add.at(self.da_trace, d, 1.0)
        self._k += 1
        if not self.enabled or self._k % self.every:
            return
        dopamine = np.maximum(self.da_trace @ self.compartment - self.da_threshold, 0.0)  # per MBON
        if not dopamine.any():
            return
        elig = self.kc_trace[self.pre] * dopamine[self.post]
        active = elig > 0
        if not active.any():
            return
        p = self.pos[active]
        w = engine.syn_weight[p]
        w0 = self.w0[active]
        dw = -self.lr * elig[active] * w0 * (self.every * self.sim.params.dt_ms)
        engine.syn_weight[p] = np.clip(w + dw, self.w_min * w0, w0)

    # --------------------------------------------------------- inspect
    def weight_ratio(self) -> np.ndarray:
        """Current / original weight of every KC->MBON synapse."""
        return self.sim.engine.syn_weight[self.pos] / self.w0

    def mbon_summary(self) -> pd.DataFrame:
        """Mean remaining KC input weight per MBON (1.0 = untouched)."""
        r = self.weight_ratio()
        w_now = np.bincount(self.post, weights=self.sim.engine.syn_weight[self.pos], minlength=len(self.mbon))
        w_orig = np.bincount(self.post, weights=self.w0, minlength=len(self.mbon))
        df = self.mbon.df[["cell_type", "side", "nt"]].copy()
        df["weight_ratio"] = np.divide(w_now, w_orig, out=np.ones_like(w_now), where=w_orig > 0)
        df["min_synapse_ratio"] = pd.Series(r).groupby(self.post).min().reindex(range(len(self.mbon))).to_numpy()
        df["plastic"] = self.plastic_mbons
        df["valence"] = df["nt"].map(VALENCE_BY_NT).fillna(0.0)
        return df

    def reset_weights(self) -> None:
        self.sim.engine.syn_weight[self.pos] = self.w0
        self.kc_trace[:] = 0
        self.da_trace[:] = 0

    def save_weights(self) -> np.ndarray:
        return self.sim.engine.syn_weight[self.pos].copy()

    def load_weights(self, w: np.ndarray) -> None:
        self.sim.engine.syn_weight[self.pos] = w


def valence_index(brain, rates: np.ndarray, mbon="MBON", softness_hz: float = 5.0) -> float:
    """Approach (+1) vs avoidance (-1) signalled by MBON activity.

    ``(approach - avoid) / (approach + avoid + softness)`` where approach /
    avoid are the summed rates of approach- and avoidance-type MBONs. The
    softness term keeps near-silent MBON populations near 0.
    """
    ns = brain[mbon]
    sign = brain.neurons["nt"].iloc[ns.idx].map(VALENCE_BY_NT).fillna(0.0).to_numpy()
    r = rates[ns.idx]
    app, av = float(r[sign > 0].sum()), float(r[sign < 0].sum())
    return (app - av) / (app + av + softness_hz)


@dataclass
class ConditioningResult:
    brain: object
    cs_plus: str
    cs_minus: str
    reinforcer: str
    mbon_rates: pd.DataFrame
    valence: pd.DataFrame
    weights: pd.DataFrame
    curve: pd.DataFrame
    kc_overlap: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def learning_index(self) -> float:
        """Change in (CS+ minus CS-) valence from pre- to post-test. >0 = learned approach."""
        v = self.valence
        pre = v.loc["pre", "CS+"] - v.loc["pre", "CS-"]
        post = v.loc["post", "CS+"] - v.loc["post", "CS-"]
        return float(post - pre)

    def summary(self) -> str:
        v = self.valence
        lines = [
            f"Olfactory conditioning: {self.cs_plus} (CS+) paired with {self.reinforcer}; {self.cs_minus} (CS-) unpaired",
            f"  KC overlap between the two odours: {self.kc_overlap:.0%}",
            f"  MBON valence  pre:  CS+ {v.loc['pre', 'CS+']:+.3f}   CS- {v.loc['pre', 'CS-']:+.3f}",
            f"  MBON valence  post: CS+ {v.loc['post', 'CS+']:+.3f}   CS- {v.loc['post', 'CS-']:+.3f}",
            f"  change:             CS+ {v.loc['post', 'CS+'] - v.loc['pre', 'CS+']:+.3f}   "
            f"CS- {v.loc['post', 'CS-'] - v.loc['pre', 'CS-']:+.3f}  (CS- change = generalisation)",
            f"  learning index: {self.learning_index:+.3f} "
            f"({'learned approach' if self.learning_index > 0.01 else 'learned avoidance' if self.learning_index < -0.01 else 'no clear memory'})",
        ]
        changed = self.weights.sort_values("weight_ratio").head(5)
        for _, r in changed.iterrows():
            lines.append(f"  {r['cell_type']:<12} ({r['side']:<5}, {r['nt']:<13}) KC input now {r['weight_ratio']:.0%} of original")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


class OlfactoryConditioning:
    """Differential olfactory conditioning on any brain with KCs, MBONs and DANs.

    ``reinforcer``: ``"sugar"``/``"bitter"`` (taste, natural route),
    ``"reward"`` (drive PAM DANs) or ``"punishment"`` (drive PPL1 DANs),
    the computational version of optogenetic memory implantation
    (Claridge-Chang et al. 2009; Aso & Rubin 2016).
    """

    def __init__(self, brain, cs_plus="banana", cs_minus="acetic_acid", reinforcer="reward",
                 concentration: float = 1.0, dan_rate_hz: float = 80.0, seed: int = 0,
                 learn_during_cs_minus: bool = False, **plasticity):
        from ..sim.simulator import Simulator

        self.brain = brain
        self.cs_plus, self.cs_minus, self.reinforcer = cs_plus, cs_minus, reinforcer
        self.conc = concentration
        self.dan_rate = dan_rate_hz
        self.learn_cs_minus = learn_during_cs_minus
        self.sim = Simulator(brain, seed=seed)
        self.plasticity = MushroomBodyPlasticity(self.sim, **plasticity)

    def _odor(self, name):
        from ..senses import smell

        return smell(self.brain, name, self.conc)

    def _reinforcement(self):
        from ..senses import taste
        from ..senses.base import Stimulus

        b = self.brain
        if self.reinforcer in ("sugar", "bitter"):
            return taste(b, self.reinforcer)
        target = {"reward": "PAM", "punishment": "PPL1"}[self.reinforcer]
        ns = b[target]
        return Stimulus(ns.idx, self.dan_rate, name=f"dopamine:{target}")

    def _present(self, stimuli, ms: float, learn: bool) -> np.ndarray:
        """One presentation from rest. Traces are cleared, so trials don't bleed into each other
        (equivalent to a long inter-trial interval)."""
        sim = self.sim
        sim.stimuli = []
        sim.add(stimuli)
        self.plasticity.enabled = learn
        self.plasticity.kc_trace[:] = 0
        self.plasticity.da_trace[:] = 0
        sim.reset()
        _, neurons = sim.advance(ms)
        sim.stimuli = []
        return np.bincount(neurons, minlength=self.brain.N) / (ms / 1000.0)

    def run(self, n_trials: int = 3, odor_ms: float = 1000.0, test_ms: float = 500.0, progress: bool = False) -> ConditioningResult:
        b = self.brain
        mb = b["MBON"]
        csp, csm = self._odor(self.cs_plus), self._odor(self.cs_minus)
        rows, curve = {}, []
        rates = {}
        rates[("pre", "CS+")] = self._present(csp, test_ms, False)
        rates[("pre", "CS-")] = self._present(csm, test_ms, False)
        kc = b["KC"].idx
        a, c = rates[("pre", "CS+")][kc] > 0, rates[("pre", "CS-")][kc] > 0
        overlap = float((a & c).sum() / max((a | c).sum(), 1))
        for k in range(n_trials):
            self._present([csp, self._reinforcement()], odor_ms, True)
            self._present(csm, odor_ms, self.learn_cs_minus)
            r = self._present(csp, test_ms, False)
            curve.append({"trial": k + 1, "valence_CS+": valence_index(b, r),
                          "kc_mbon_weight": float(self.plasticity.weight_ratio().mean())})
            if progress:
                print(f"  training trial {k + 1}/{n_trials}: CS+ valence {curve[-1]['valence_CS+']:+.3f}", flush=True)
        rates[("post", "CS+")] = self._present(csp, test_ms, False)
        rates[("post", "CS-")] = self._present(csm, test_ms, False)
        for key, r in rates.items():
            rows[key] = r[mb.idx]
        table = pd.DataFrame(rows, index=mb.idx)
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        table.insert(0, "cell_type", mb.df["cell_type"].to_numpy())
        table.insert(1, "nt", mb.df["nt"].to_numpy())
        val = pd.DataFrame(
            {cs: {phase: valence_index(b, rates[(phase, cs)]) for phase in ("pre", "post")} for cs in ("CS+", "CS-")}
        )
        notes = []
        if not a.any():
            notes.append(f"{self.cs_plus} activated no Kenyon cells; try a higher concentration")
        return ConditioningResult(b, self.cs_plus, self.cs_minus, self.reinforcer, table, val,
                                  self.plasticity.mbon_summary(), pd.DataFrame(curve), overlap, notes)
