"""Compare openfly with the published Brian2 reference simulation of Shiu et al. 2024.

Downloads the FlyWire v630 connectivity used in the paper and the reference
spikes (results/example/sugarR.parquet: 21 right sugar GRNs at 200 Hz,
30 trials of 1 s), runs the same experiment in openfly and reports the
per-neuron firing-rate correlation (expect r > 0.999).
"""

import sys

import numpy as np
import pandas as pd

import openfly as of
from openfly.data.download import fetch
from openfly.data.registry import SHIU_REF
from openfly import config

SUGAR_630 = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
             720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
             720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
             720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
MN9_630 = 720575940660219265

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
brain = of.load("flywire_630")
ref_path = fetch(f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_REF}/results/example/sugarR.parquet",
                 config.raw_dir("flywire_630") / "sugarR.parquet")
ref = pd.read_parquet(ref_path)
ref_rate = ref.groupby("flywire_id").size() / ref.trial.nunique()

sim = brain.simulator(seed=0)
sim.stimulate(brain.from_ids(SUGAR_630), 200.0)
res = sim.run(1000, n_trials=n_trials, progress=True)
mine = pd.Series(res.rates(), index=brain.neurons.root_id)
both = pd.concat([ref_rate.rename("reference"), mine.rename("openfly")], axis=1).fillna(0)
active = both[(both.reference > 0) | (both.openfly > 0)]
print(f"spikes per trial: reference {len(ref) / ref.trial.nunique():,.0f}, openfly {res.n_spikes / n_trials:,.0f}")
print(f"per-neuron rate correlation r = {np.corrcoef(active.reference, active.openfly)[0, 1]:.4f}")
print(f"MN9: reference {ref_rate.get(MN9_630, 0):.1f} Hz, openfly {mine.get(MN9_630, 0):.1f} Hz")
