"""Model parameters for the whole-brain leaky integrate-and-fire (LIF) model.

The defaults reproduce the model of Shiu et al. (2024), "A Drosophila
computational brain model reveals sensorimotor processing", Nature 634.
Every constant carries the reference it was taken from, so you can see
which numbers are measured biology and which are free parameters.

All values are stored as plain floats in **milliseconds** and **millivolts**
(no unit library needed). See ``docs/05-simulation.md`` for the equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class LIFParams:
    """Constants of the LIF network.

    Membrane equations (between spikes, for every neuron)::

        dv/dt = (v_rest - v + g) / tau_m
        dg/dt = -g / tau_syn

    A presynaptic spike adds ``w_syn * (signed synapse count)`` to ``g`` of
    each postsynaptic partner after ``delay_ms``. When ``v > v_thresh`` the
    neuron spikes, ``v`` resets to ``v_reset``, ``g`` is cleared and the
    neuron is refractory for ``t_refrac_ms``.
    """

    # Kakaria & de Bivort 2017, doi:10.3389/fnbeh.2017.00008
    v_rest: float = -52.0  # mV, resting potential
    v_reset: float = -52.0  # mV, potential after a spike
    v_thresh: float = -45.0  # mV, spike threshold
    tau_m: float = 20.0  # ms, membrane time constant

    # Jürgensen et al. 2021, doi:10.1088/2634-4386/ac3ba6
    tau_syn: float = 5.0  # ms, synaptic conductance decay

    # Lazar et al. 2021, doi:10.7554/eLife.62362
    t_refrac_ms: float = 2.2  # ms, refractory period

    # Paul et al. 2015, doi:10.3389/fncel.2015.00029
    delay_ms: float = 1.8  # ms, synaptic delay

    # The single free parameter of the Shiu model, fitted to behaviour.
    w_syn: float = 0.275  # mV per synapse

    # Poisson "optogenetic" drive: each input event kicks v by
    # w_syn * poisson_scale (250 * 0.275 = 68.75 mV) -> forces a spike.
    poisson_scale: float = 250.0

    # Integration step. Brian2's default, used by the original model.
    dt_ms: float = 0.1

    # In the original Brian2 model, ``g`` is flagged "(unless refractory)",
    # and synaptic input that reaches a neuron while it is refractory is
    # lost. We reproduce that by default (it was verified against Brian2
    # 2.10, see docs/13-validation.md). Set to False to keep that input,
    # which makes strongly driven neurons fire ~20-30% faster.
    drop_input_when_refractory: bool = True

    def with_(self, **changes) -> "LIFParams":
        """Return a copy with some fields changed: ``p.with_(w_syn=0.3)``."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def poisson_kick_mv(self) -> float:
        return self.w_syn * self.poisson_scale

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.delay_ms / self.dt_ms)))

    @property
    def refrac_steps(self) -> int:
        return int(round(self.t_refrac_ms / self.dt_ms))


#: Parameters of Shiu et al. 2024 (the published whole-brain model).
SHIU_2024 = LIFParams()


@dataclass
class TrialConfig:
    """How long and how often to run an experiment."""

    duration_ms: float = 1000.0
    n_trials: int = 1
    seed: int | None = 0
    extra: dict = field(default_factory=dict)
