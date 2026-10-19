# 15. Build journal: how openfly was made

A chronological account of building openfly: what was done, why, what
went wrong and how it was fixed. It is a record, and a lesson in how to
work with connectome data.

## 1. Finding the brain file

* The FlyWire connectome is public, but the obvious portals (Zenodo,
  Codex web UI) were not reachable from the build machine. Two Git
  repositories were: the **Shiu et al. 2024 model** repo (connectivity
  table for v630 and v783, plus reference simulation output) and
  **flyconnectome/flywire_annotations** (cell types, positions,
  transmitters for all 139k neurons).
* The FlyWire Codex **public storage bucket** turned out to be listable and
  to hold every Codex dataset in one shared table layout: FAFB, BANC, MANC,
  male CNS, male optic lobe. That became the generic adapter that lets
  openfly load future releases by name (`codex:<dataset>/<version>`).
* Decision: download at runtime from the original publishers (pinned
  commits), never redistribute data, and compile a local `.flybrain` cache.

## 2. A simulator you can step

* The published model is Brian2 code that builds and runs a whole network per
  trial. For a live playground, a virtual body and learning rules, we needed a
  simulator advanced in small steps with hooks, so we re-implemented
  it: exact integration, Brian2's operation order, an **awake set** so resting
  neurons cost nothing.
* **First validation failed**: r = 0.97 with the reference, but 24% too many spikes.
* Debugging: Brian2 was installed and run on the same data (126 s per trial),
  then compared step by step on a 2-neuron network. Finding: **Brian2
  discards synaptic input that arrives during the refractory period**
  ("unless refractory" on `g`). Reproducing it gave r = 0.9996.
* A numba kernel added ~3x speed. A parallel version was *slower* (thread
  overhead at 10,000 steps per simulated second) and was dropped, as was sorting
  the awake list (also slower).

## 3. Data model and names

* One schema for every brain: a neuron table + signed sparse matrix; a
  portable `.flybrain` ZIP of Parquet files.
* A curated alias vocabulary (MN9 = `CB0701`, giant fiber = `DNp01`...),
  checked against the annotation table. For example, MN9 had to be found
  through the root ID used in the Shiu paper.

## 4. The toy brain, and calibrating it

* An offline ~2,300-neuron cartoon with real names, for lessons and tests.
* First version: nothing propagated. We measured the model's
  input-output curve (summed rate x synapses -> output rate) and rewired
  with that table: silent below ~2,500, ~44 Hz at 10,000, ~81 Hz at 23,000.
* Vision bug: full-field flashes triggered escapes. Fix: a CT1-like wide-field
  GABAergic neuron, fed directly from photoreceptors so its inhibition arrives
  in step with the excitation.

## 5. The real brain surprised us

* Sugar -> MN9 worked at once. **Smell exploded**: any odour activated
  ~8,000 neurons, with antennal-lobe local neurons at 260 Hz.
* Tracing the first spikes pointed to local neurons predicted as
  serotonergic, dopaminergic or cholinergic (so excitatory in the model),
  although most are GABAergic in real flies. We added **transmitter
  policies** (published vs curated) instead of silently changing the data.
  The curated policy gives sparse Kenyon-cell codes (~3.5%).

## 6. Learning on the connectome

* The mushroom-body rule was written so that compartments come from
  DAN -> MBON wiring. The data reproduced the textbook map.
* First FlyWire runs learned the wrong thing: reward weakened *approach*
  MBONs. Causes: (1) compartment membership from any single synapse, fixed by
  synapse-weighted dopamine; (2) odours alone excite DANs in the whole-brain
  model, fixed with a dopamine threshold and no learning during CS-
  presentations; (3) odours sharing 33% of Kenyon cells generalise, so we chose
  banana vs acetic acid (3% overlap). Result: odour-specific memories of both signs.

## 7. Screens and training

* A rate-model approximation (fitted to the LIF curve, r = 0.96-0.98)
  makes screens of hundreds of cell types take minutes. The activation +
  silencing screens singled out **CB0553** as necessary for sugar-evoked MN9.
* Evolution strategies tune cell-type gains; the optimiser rediscovered "make
  the bitter pathway stronger" to reject laced sugar.

## 8. A body for the brain

* First virtual fly: found the banana, then **orbited** it at 20 mm.
  Diagnosis by stationary tests: the left/right signal was literally zero
  (receptor model at 20% activation), then tiny (3% concentration
  differences flattened by saturation).
* Fixes, each biologically motivated: Weber-law receptor adaptation,
  run-and-turn on lateral-horn activity changes (klinokinesis), bilateral
  mutual inhibition in the toy LAL, slowing while turning, stopping on food
  contact, and capping the visual size of static food so approaching a fruit
  isn't mistaken for a predator.
* Result on the toy: 6/6 foraging, 6/6 escapes, never eats bitter bait.
* **Then a check that humbled it**: those runs started facing the banana.
  Started facing *away*, the fly found it in only 1 of 3 runs within 12 s.
  So we built a fair benchmark: 10 random starts, 30 s, intact vs. smell
  neurons silenced vs. smell and vision silenced. Result: **10/10 vs 3/10 vs
  2/10**. Odour guidance really comes from the circuit, and the honest
  number is in the docs, not the flattering one.
* FlyWire closed loop: escape works (20 ms after the attack). From the
  default start, 82 mm away, smell-guided walking seemed absent.

## 9. Reading intentions: a population decoder

* Idea from brain-computer interfaces: instead of one steering neuron,
  *decode* turning intent from many. First tries failed to generalise: the
  decoder confused "weaker smell" with "smell on the other side". Ridge
  regression with symmetric conditions at several intensities, reading the
  premotor steering regions (LAL/SPS/IPS, 168 responsive neurons including
  PFL3), decoded held-out conditions with the correct sign (-0.19 / +0.21 / -0.06).
  See [09](09-virtual-fly.md#a-population-decoder-for-real-brains).
* In closed loop the first number looked bad: a bearing-turn correlation
  of +0.05. Then a sign check: turn > 0 means *right*, bearing > 0 means
  food on the *left*, so correct steering is a *negative* correlation, and
  +0.05 meant "none", not "weak".
* The decisive test was a control, not a better decoder. Started facing
  away, the FlyWire fly turned round and cut its distance by 40-50% with smell
  (with either readout, and with klinokinesis off: correlations -0.11 to
  -0.17), and never approached with its ORNs silenced. Smell steering is
  in the connectome, but weak, and the decoder was no better than
  high-passed DNa02. See [09](09-virtual-fly.md#on-the-real-flywire-brain).

## 10. Playground, docs, tests

* Front end with no build step and no CDN: one WebGL2 point cloud whose
  shader turns spike times into glow, Server-Sent Events from a
  standard-library server, canvas charts, Web Audio sonification.
* Visual QA with headless Chromium screenshots led to a few fixes:
  idle neurons too faint, ripple map too dark (log scale + brighter base).
* 64 offline tests, 5 FlyWire tests, every lesson snippet executed
  automatically against the toy brain.
