# 7. Behaviour readout

Code: `openfly/motor.py`

The brain talks to the body through ~1,300 descending neurons (DNs) and
~110 head motor neurons. A few are understood well enough to read as
"the fly wants to do X". A `MotorChannel` names such a group and turns its
firing rate into a **drive**:

* `mode="mean"`: `drive = r / (r + r_half)`, from 0 (silent) to approaching 1
  (`r_half` = rate that counts as half-maximal);
* `mode="lr"`: `drive = (R - L) / (R + L + r_half)` from -1 (left) to +1 (right).

| channel | neurons | meaning | evidence |
|---|---|---|---|
| `feeding` | MN9 (`CB0701`) | proboscis extension | Shiu et al. 2024; Gordon & Scott 2009 |
| `walk_forward` | DNp09 ("P9") | forward walking | Bidaye et al. 2020 (*likely*) |
| `walk_backward` | MDN | backward walking | Bidaye et al. 2014 |
| `turn` | DNa02 left vs right | steering | Rayshubskiy et al. 2020; Yang et al. 2024 |
| `escape` | giant fiber (DNp01) | take-off jump | von Reyn et al. 2014 |
| `receptivity` | pC1 | female receptivity / social arousal | Zhou et al. 2014 (*likely*) |
| `grooming` | `groom_DN` | toy brain only | |

```python
res = sim.run(500)
res.behavior()
# {'feeding': {'drive': 0.78, 'rate_hz': 69.0, 'left_hz': 66.0, 'right_hz': 72.0, ...}, ...}
print(openfly.motor.describe(res.behavior()))
# feeding       ########   0.78  (69 Hz)
# turn                     +0.00  straight (L 0 / R 0 Hz)
```

**Live readout.** `MotorMonitor(brain).update(spikes, window_ms)` returns
exponentially smoothed drives (tau = 100 ms). The playground meters and the
virtual fly use it.

**Add a channel.**

```python
from openfly.motor import MotorChannel, add_channel
add_channel(MotorChannel("landing", "DNp10", "landing response", "my notes", half_rate_hz=15))
```

**Caveat.** Real behaviour is a *population code* across many DNs (Braun
et al. 2024). A single DN type is a useful, simplified indicator, not the
full story. When the readout neuron isn't recruited by a stimulus in the
model, that is a prediction to check, not proof the fly wouldn't move.
