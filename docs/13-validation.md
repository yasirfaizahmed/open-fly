# 13. Validation: how we know the simulator is right

A brain simulator is only useful if it reproduces the model it claims to
implement. openfly's engine was checked against the **original Brian2
code** of Shiu et al. (2024), run in this project's environment.

## 1. Against the published reference simulation

The Shiu et al. repository ships the spikes of a reference experiment
(`results/example/sugarR.parquet`): 21 right-side sugar GRNs driven at
200 Hz, FlyWire v630, 30 trials of 1 s. openfly ran the same experiment
on the same connectivity files:

| | reference (Brian2) | openfly NumPy | openfly numba |
|---|---|---|---|
| spikes per trial | 17,052 | 17,108 (5-trial mean) | 16,748 |
| neurons active | 448 (30 trials) | 428 (5 trials) | 431 (5 trials) |
| **per-neuron rate correlation** | | **r = 0.9996** | **r = 0.9994** |
| MN9 rate | 93.3 Hz | 91.0 Hz | 92.2 Hz |
| time per 1 s trial | ~126 s (Brian2 C++, incl. compile) | ~5.6 s | ~1.7-2.7 s |

Fewer "active" neurons comes from fewer trials (rarely-firing neurons are
missed), not from the model.

## 2. The refractory-input detail

A first version of the engine gave **24% too many spikes** (21,100 vs
17,052 per trial) despite r = 0.97. Comparing the two simulators step by
step on a 2-neuron network found the cause. In Brian2, the synaptic variable
`g` is marked "(unless refractory)", and synaptic input reaching a neuron
*during its refractory period* is discarded (we observed `g` staying 0 after
an input arrived at a refractory neuron, while a non-refractory control
neuron received it). Reproducing that rule brought the spike count to
17,108 and the correlation to 0.9996.

This rule is `LIFParams.drop_input_when_refractory` (default `True`).
`tests/test_engine.py::test_input_during_refractory_is_dropped` pins it down
with a deterministic 2-neuron test.

## 3. Unit tests

`pytest` runs 63 tests in ~15 s on the toy brain, plus 5 optional
FlyWire checks (`OPENFLY_TEST_FLYWIRE=1`):

* engine: Poisson drive rate, synapse-count monotonicity, **exact
  integration vs the closed-form solution**, refractory input dropping,
  silencing, seeds, current injection, numba == numpy statistically;
* brain API, `.flybrain` round trip, BrainBuilder, transmitter policies;
* adapters on synthetic files in the real FlyWire / Codex / neuPrint formats;
* senses (laterality, dynamics, scenarios), simulator, results, rate model;
* learning (conditioning sign, ES improvement, screens), graph tools, virtual fly;
* playground over real HTTP + Server-Sent Events.

FlyWire checks: neuron/connection counts (138,639 / 15,091,983), sugar ->
MN9 > 20 Hz, looming -> giant fiber > 20 Hz, sparse KC code under the
curated policy, strongest path ends at MN9.

## 4. Biological sanity checks on FlyWire

| experiment | model | expected from biology |
|---|---|---|
| sugar GRNs 10 / 25 / 50 / 100 / 150 / 200 Hz | MN9 0 / 0 / 31 / 57 / 80 / 81 Hz | threshold then saturation (Shiu et al. 2024) |
| sugar + bitter | MN9 0 Hz | bitter suppresses feeding |
| looming at +50 deg | giant fiber 64 (left) / 108 (right) Hz | ipsilateral-dominant escape |
| banana odour (curated) | ~3.5% of KCs active | 5-10% (Turner et al. 2008; Honegger et al. 2011) |
| conditioning (reward / punishment) | learning index +0.27 / -0.21, odour-specific | approach / avoidance memory |

## Reproduce it

```bash
pip install brian2
python examples/validate_against_brian2.py      # downloads v630 data, compares with sugarR.parquet
```
