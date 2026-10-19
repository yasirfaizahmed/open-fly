# 14. Findings and limitations

## Things exploring FlyWire with openfly turned up

These came out of building and testing the SDK. They are model results,
not established biology. Treat them as hypotheses, and as examples of what
the tools can do.

1. **The antennal lobe explodes under the published signs.** About 130
   antennal-lobe local neurons are predicted serotonergic, dopaminergic or
   cholinergic, so they excite. Any odour then triggers runaway activity
   (~8,000 neurons, LNs at ~260 Hz). Re-signing them as GABAergic, which
   most are in real flies, restores a sparse odour code (~3.5% of Kenyon
   cells). The `"curated"` transmitter policy exists because of this.
   ([05](05-simulation.md#transmitter-policies-when-predictions-go-wrong))
2. **CB0553 looks necessary for sugar-evoked feeding.** An activation screen
   ranks it as the strongest driver of MN9; silencing it drops
   sugar-evoked MN9 firing from ~80 to ~4 Hz.
3. **Sugar -> MN9 has a sharp threshold.** No response up to 25 Hz of sugar
   GRN input, 31 Hz at 50 Hz input, saturating at ~80 Hz.
4. **Mushroom-body compartments are written in the wiring.** DAN -> MBON
   synapse counts alone recover the classic compartment map (PPL1-gamma1 ->
   MBON11, PAM -> MBON01-09, PPL1 -> MBON11-19), which is enough to run
   odour-specific reward and punishment learning.
5. **Sugar taste does not reach reward dopamine neurons** in the LIF model,
   although sugar is the classic reward. Either the pathway runs through
   neuromodulation or the VNC, outside this model, or point neurons are too
   simple to carry it.
6. **Steering readout has a left bias under smell.** Left DNa02 fires at
   37.5 Hz when banana is on the left and 15 Hz when it is on the right, while
   right DNa02 stays near 0. So the absolute readout always says "turn left",
   but its *changes* track the odour side, which bias removal (high-pass) can use.
7. **Odour-guided navigation does not (yet) emerge on FlyWire.** In closed
   loop the FlyWire virtual fly does not find the banana in 12 s: odour barely
   recruits the walking/steering DNs used as readouts. Escape does work: a
   predator attack triggers the giant-fiber jump 20 ms later.
   ([09](09-virtual-fly.md#on-the-real-flywire-brain))
8. **Cross-dataset transfer works.** Transmitters and classes borrowed from
   FlyWire make the hemibrain simulate sensibly: looming drives its giant
   fiber to 162 Hz.

## Limitations: read before drawing conclusions

**Model**
* Point neurons with identical parameters; no dendrites, graded potentials,
  gap junctions, neuromodulation or plasticity (except the learning rule
  you switch on). Early vision (graded, timing-sensitive) is the weakest
  part, hence the feature-level vision stimuli.
* Synaptic weight = synapse count x 0.275 mV, with sign from predicted transmitters.
* Dopamine, serotonin and octopamine are treated as fast excitatory
  transmitters, as in the published model.
* No spontaneous activity: a brain with no input is silent.

**Data**
* One female brain (FlyWire), with individual variability and proofreading errors.
* Transmitter predictions are imperfect; see the curated policy.
* Retinotopy is estimated from anatomy.
* Codex releases without coordinates use a schematic layout.

**Interfaces**
* The odour table is simplified; use DoOR data for research.
* Behaviour readouts use single DN types as indicators of what are really
  population codes.
* The virtual fly is a point body with an exploration drive and a body-level
  run-and-turn rule; both are documented and switchable.

**Speed**
* Full-brain simulation is slower than real time for strong visual stimuli
  (0.05-0.2x); taste and smell run at ~0.2-0.6x with numba.
