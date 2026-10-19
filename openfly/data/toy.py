"""The "toy fly": a small synthetic brain for learning and offline use.

The real FlyWire brain has ~139,000 neurons and is a 130 MB download.
The toy fly has ~2,500 neurons, builds in about a second and needs no
internet. It is used by the tutorials, the tests and the playground's
offline mode.

It is **not** real data. It is a cartoon wired by hand, using real cell
type *names* and textbook pathways, so everything you learn transfers:

* smell: ORNs (10 glomeruli) -> projection neurons -> lateral horn (innate)
  and mushroom body (learned); bilateral input lets the fly steer
* taste: sugar/water GRNs excite MN9 (feeding); bitter/salt GRNs inhibit it
* vision: photoreceptors with a retinotopic map -> L2 -> LC4/LPLC2
  (looming) -> giant fiber escape; LC10 small objects -> approach
* learning: KCs -> MBONs with dopaminergic PAM (reward) and PPL1
  (punishment) neurons defining compartments, APL feedback inhibition
* hearing, wind, touch, heat and humidity pathways
* a few hundred "background" central neurons for brain-wide activity

Run ``openfly.load("toy")`` to get it. See ``docs/03-data-and-brain-files.md``.
"""

from __future__ import annotations

import numpy as np

from .builder import BrainBuilder, group
from .layout import region_centers

GLOMERULI = ["DM1", "DM2", "DM3", "DM4", "VA2", "DP1l", "DC1", "DA2", "V", "DA1"]
ATTRACTIVE = ["DM1", "DM2", "DM3", "DM4", "VA2", "DP1l", "DC1"]
AVERSIVE = ["DA2", "V"]

_C = region_centers()


def _c(region: str, side: str):
    key = f"{region}_{'L' if side == 'left' else 'R'}"
    return _C.get(key, _C.get(region))


def _retina(n_az: int, n_el: int, side: str):
    """Grid of (azimuth, elevation) in degrees + positions on a lamina-like sheet."""
    sgn = 1 if side == "right" else -1
    az = np.linspace(-10, 160, n_az)
    el = np.linspace(-55, 55, n_el)
    A, E = np.meshgrid(az, el)
    A, E = A.ravel(), E.ravel()
    cx, cy, cz = _c("LA", side)
    pos = np.stack([cx + sgn * 25 * np.cos(np.radians(A - 75)), cy - 1.6 * E, cz + 1.1 * (A - 75)], axis=1)
    return sgn * A, E, pos


def make_toy_brain(seed: int = 7):
    b = BrainBuilder("toy", seed=seed, title="Toy fly (synthetic teaching brain)",
                     species="Drosophila melanogaster (cartoon)",
                     description="Hand-wired ~2,500-neuron brain using real cell-type names. Not real data.",
                     citation="openfly toy brain (synthetic)", license="MIT")
    sides = ("left", "right")
    other = {"left": "right", "right": "left"}
    P: dict[tuple[str, str], object] = {}

    def add(key, side, cell_type, n, **kw):
        kw.setdefault("side", side)
        P[(key, side)] = b.add(cell_type, n, **kw)
        return P[(key, side)]

    for s in sides:
        # ---------------------------------------------------------- smell
        for g in GLOMERULI:
            add(f"ORN_{g}", s, f"ORN_{g}", 12, super_class="sensory", cell_class="olfactory", region="AL",
                center=_c("AL", s), spread=9, nt="acetylcholine",
                cell_sub_class="pheromone" if g == "DA1" else "")
            add(f"PN_{g}", s, f"{g}_adPN", 3, cell_class="ALPN", cell_sub_class="uniglomerular", region="AL",
                center=np.add(_c("AL", s), (0, -20, 25)), spread=10)
        add("LN", s, "lLN2F", 6, cell_class="ALLN", nt="gaba", region="AL", center=_c("AL", s), spread=12)
        add("KC", s, "KCg-m", 300, cell_class="Kenyon_Cell", region="MB_CA", center=_c("MB_CA", s), spread=14)
        add("APL", s, "APL", 1, cell_class="MBIN", nt="gaba", region="MB_VL", center=_c("MB_VL", s))
        add("MBON01", s, "MBON01", 1, cell_class="MBON", nt="glutamate", region="MB_ML", center=_c("MB_ML", s))
        add("MBON03", s, "MBON03", 1, cell_class="MBON", nt="glutamate", region="MB_ML", center=_c("MB_ML", s))
        add("MBON11", s, "MBON11", 1, cell_class="MBON", nt="gaba", region="MB_ML", center=_c("MB_ML", s))
        add("MBON12", s, "MBON12", 1, cell_class="MBON", nt="acetylcholine", region="MB_VL", center=_c("MB_VL", s))
        add("PAM", s, "PAM08", 3, cell_class="DAN", nt="dopamine", region="MB_ML", center=np.add(_c("MB_ML", s), (0, 15, -25)))
        add("PPL1", s, "PPL101", 2, cell_class="DAN", nt="dopamine", region="MB_VL", center=np.add(_c("MB_VL", s), (0, -10, 25)))
        add("LH_att", s, "LHAV1a3", 4, cell_class="LHLN", region="LH", center=_c("LH", s), spread=8)
        add("LH_av", s, "LHAV4a4", 4, cell_class="LHLN", region="LH", center=_c("LH", s), spread=8)
        add("LAL_app", s, "LAL_app", 2, region="LAL", center=_c("LAL", s))
        add("LAL_av", s, "LAL_av", 2, region="LAL", center=_c("LAL", s))
        # ---------------------------------------------------------- motor
        for key, ct, region in (("DNa02", "DNa02", "LAL"), ("DNa01", "DNa01", "LAL"), ("DNp09", "DNp09", "SPS"),
                                ("MDN", "MDN", "VES"), ("GF", "DNp01", "PVLP"), ("groomDN", "groom_DN", "GNG")):
            add(key, s, ct, 2 if key == "MDN" else 1, super_class="descending", region=region,
                center=_c(region, s), spread=5)
        add("MN9", s, "CB0701", 1, super_class="motor", cell_class="brain_motor_neuron",
            cell_sub_class="ingestion_motor_neuron", region="GNG", center=np.add(_c("GNG", s), (0, 30, -40)))
        # ---------------------------------------------------------- taste
        gc = np.add(_c("GNG", s), (0, 25, -30))
        add("sugar", s, "LB3c", 8, super_class="sensory", cell_class="gustatory", cell_sub_class="sugar", region="GNG", center=gc)
        add("water", s, "LB3a", 6, super_class="sensory", cell_class="gustatory", cell_sub_class="water", region="GNG", center=gc)
        add("bitter", s, "LB1a", 8, super_class="sensory", cell_class="gustatory", cell_sub_class="bitter", region="GNG", center=gc)
        add("salt", s, "LB3d", 4, super_class="sensory", cell_class="gustatory",
            cell_sub_class="high_salt/heavy_metal", region="GNG", center=gc)
        add("G2N", s, "G2N-1", 3, region="GNG", center=_c("GNG", s))
        add("bitterIN", s, "GABA_bitter_IN", 2, nt="gaba", region="GNG", center=_c("GNG", s))
        # --------------------------------------------------------- vision
        az, el, pos = _retina(10, 8, s)
        add("R16", s, "R1-6", len(az), super_class="sensory", cell_class="visual", region="LA", positions=pos,
            azimuth=az, elevation=el)
        az7, el7, pos7 = _retina(5, 4, s)
        add("R7", s, "R7", len(az7), super_class="sensory", cell_class="visual", region="ME",
            positions=pos7 + np.array([(-1 if s == "left" else 1) * 90, 0, 0]), azimuth=az7, elevation=el7)
        add("R8", s, "R8", len(az7), super_class="sensory", cell_class="visual", region="ME",
            positions=pos7 + np.array([(-1 if s == "left" else 1) * 95, 4, 0]), azimuth=az7, elevation=el7)
        add("L2", s, "L2", len(az), super_class="optic", cell_class="LA>ME", region="LA",
            positions=pos + np.array([(-1 if s == "left" else 1) * -40, 0, 0]), azimuth=az, elevation=el)
        az5, el5, pos5 = _retina(5, 4, s)
        for d in "abcd":
            for kind, dx in (("T4", -150), ("T5", -170)):
                add(f"{kind}{d}", s, f"{kind}{d}", len(az5), super_class="optic", cell_class="ME>LOP", region="LOP",
                    positions=pos5 + np.array([(-1 if s == "left" else 1) * dx, 10, 60]), azimuth=az5, elevation=el5)
        lo = _c("LO", s)
        az4 = np.linspace(-5, 150, 8) * (1 if s == "right" else -1)
        add("LC4", s, "LC4", 8, super_class="visual_projection", region="LO", center=lo, azimuth=az4,
            elevation=np.tile([-20, 20], 4))
        add("LPLC2", s, "LPLC2", 8, super_class="visual_projection", region="LOP", center=_c("LOP", s),
            azimuth=az4, elevation=np.tile([20, -20], 4))
        add("LC10", s, "LC10a", 6, super_class="visual_projection", region="LO", center=lo,
            azimuth=np.linspace(-5, 120, 6) * (1 if s == "right" else -1), elevation=np.zeros(6))
        add("CT1", s, "CT1", 1, super_class="optic", cell_class="ME.LO", nt="gaba", region="LO", center=lo)
        add("HS", s, "HSE", 1, super_class="visual_projection", region="LOP", center=_c("LOP", s))
        add("VS", s, "VS1", 1, super_class="visual_projection", region="LOP", center=_c("LOP", s))
        # ---------------------------------------------------- mechanosense
        ammc = _c("AMMC", s)
        add("JO_aud", s, "JO-B", 8, super_class="sensory", cell_class="mechanosensory", cell_sub_class="auditory",
            region="AMMC", center=ammc)
        add("JO_wind", s, "JO-CE", 8, super_class="sensory", cell_class="mechanosensory",
            cell_sub_class="wind_gravity", region="AMMC", center=ammc)
        add("AMMC_B", s, "AMMC-B1", 2, region="AMMC", center=ammc)
        add("AMMC_A", s, "AMMC-A2", 2, region="AMMC", center=ammc)
        add("pC1", s, "pC1a", 1, region="SMP", center=_c("SMP", s))
        add("WED", s, "WED_song", 1, nt="gaba", region="WED", center=_c("WED", s))
        add("bristle", s, "BM_InOm", 6, super_class="sensory", cell_class="mechanosensory",
            cell_sub_class="eye bristle", region="GNG", center=np.add(_c("GNG", s), (0, -20, -30)))
        add("groomIN", s, "groom_IN", 2, region="GNG", center=_c("GNG", s))
        # ------------------------------------------------- thermo / hygro
        add("hot", s, "TRN_VP2", 3, super_class="sensory", cell_class="thermosensory", region="AL", center=_c("AL", s))
        add("hotPN", s, "VP2_adPN", 1, cell_class="ALPN", region="LH", center=_c("LH", s))
        add("moist", s, "HRN_VP1d", 3, super_class="sensory", cell_class="hygrosensory", region="AL", center=_c("AL", s))
        add("moistPN", s, "VP1d_il2PN", 1, cell_class="ALPN", region="LH", center=_c("LH", s))

    bg = b.add("CB_toy", 400, region="SMP", center=(528, 200, 140), spread=70, nt="acetylcholine")
    bg_inh = bg[320:]
    for k in range(bg_inh.start, bg_inh.stop):
        b._rows[k]["nt"] = "gaba"

    # =============================================================== wiring
    # Synapse counts are calibrated to the Shiu LIF model: a neuron needs a
    # summed drive (rate x synapses over its inputs) of ~3,000 Hz*syn to
    # start firing and ~12,000 Hz*syn to fire at ~50 Hz. See
    # docs/03-data-and-brain-files.md ("calibrating a hand-made brain").
    c = b.connect
    for s in sides:
        o = other[s]
        # smell: ORNs project to both antennal lobes, ipsilateral stronger
        for g in GLOMERULI:
            c(P[(f"ORN_{g}", s)], P[(f"PN_{g}", s)], synapses=(5, 7))
            c(P[(f"ORN_{g}", s)], P[(f"PN_{g}", o)], synapses=(2, 4))
            c(P[(f"ORN_{g}", s)], P[("LN", s)], p=0.3, synapses=2)
            c(P[("LN", s)], P[(f"PN_{g}", s)], synapses=3)
            if g in ATTRACTIVE:
                c(P[(f"PN_{g}", s)], P[("LH_att", s)], synapses=(40, 50))
            elif g in AVERSIVE:
                c(P[(f"PN_{g}", s)], P[("LH_av", s)], synapses=(55, 65))
        pn_all = group(*[P[(f"PN_{g}", s)] for g in GLOMERULI], name="PN")
        kc = P[("KC", s)]
        c(pn_all, kc, n_inputs=6, synapses=(25, 35))
        c(kc, P[("APL", s)], synapses=10)
        c(P[("APL", s)], kc, synapses=120)
        for m in ("MBON01", "MBON03", "MBON11", "MBON12"):
            c(kc, P[(m, s)], p=0.5, synapses=(15, 25))
        for m in ("MBON01", "MBON03"):  # reward (PAM) compartments
            c(P[("PAM", s)], P[(m, s)], synapses=1)
        for m in ("MBON11", "MBON12"):  # punishment (PPL1) compartments
            c(P[("PPL1", s)], P[(m, s)], synapses=1)
        c(P[("MBON12", s)], P[("LAL_app", s)], synapses=40)
        c(P[("MBON11", s)], P[("LAL_av", s)], synapses=40)  # GABA: learned approach suppresses avoidance
        c(P[("MBON01", s)], P[("LAL_app", s)], synapses=40)  # glutamate: inhibits approach
        c(P[("MBON03", s)], P[("LAL_app", s)], synapses=40)
        # innate steering: approach -> ipsilateral turn, avoid -> contralateral turn + back up
        c(P[("LH_att", s)], P[("LAL_app", s)], synapses=60)
        c(P[("LH_av", s)], P[("LAL_av", s)], synapses=60)
        c(P[("LAL_app", s)], P[("DNa02", s)], synapses=100)
        c(P[("LAL_app", s)], P[("DNp09", s)], synapses=80)
        c(P[("LAL_av", s)], P[("DNa02", o)], synapses=100)
        c(P[("LAL_av", s)], P[("MDN", s)], synapses=80)
        c(P[("LAL_av", s)], P[("MDN", o)], synapses=60)
        # taste: sugar/water excite MN9, bitter/salt inhibit it; sugar is a reward, bitter a punishment
        c(P[("sugar", s)], P[("G2N", s)], synapses=16)
        c(P[("water", s)], P[("G2N", s)], synapses=14)
        c(P[("bitter", s)], P[("bitterIN", s)], synapses=16)
        c(P[("salt", s)], P[("bitterIN", s)], synapses=24)
        c(P[("G2N", s)], P[("MN9", s)], synapses=45)
        c(P[("G2N", s)], P[("MN9", o)], synapses=29)
        c(P[("bitterIN", s)], P[("MN9", s)], synapses=60)
        c(P[("bitterIN", s)], P[("MN9", o)], synapses=40)
        c(P[("G2N", s)], P[("PAM", s)], synapses=45)
        c(P[("G2N", s)], P[("PAM", o)], synapses=29)
        c(P[("bitter", s)], P[("PPL1", s)], synapses=12)
        c(P[("bitter", s)], P[("MDN", s)], synapses=6)
        # vision: retinotopic photoreceptor -> L2 -> feature detectors -> DNs
        c(P[("R16", s)], P[("L2", s)], one_to_one=True, synapses=80)
        b.connect_by_distance(P[("L2", s)], P[("LC4", s)], radius=40, synapses=30, key=("azimuth", "elevation"))
        b.connect_by_distance(P[("L2", s)], P[("LC10", s)], radius=22, synapses=100, key=("azimuth", "elevation"))
        for d in "abcd":
            for kind in ("T4", "T5"):
                t = P[(f"{kind}{d}", s)]
                b.connect_by_distance(P[("L2", s)], t, radius=20, synapses=2, key=("azimuth", "elevation"))
                b.connect_by_distance(t, P[("LPLC2", s)], radius=45, synapses=1, key=("azimuth", "elevation"))
                c(t, P[("HS", s)] if d in "ab" else P[("VS", s)], synapses=6)
        # wide-field GABAergic neuron: full-field flashes/flicker suppress the
        # object detectors, so only local, expanding things look like a threat
        # (fed straight from the photoreceptors so the inhibition arrives in
        # step with the L2 excitation)
        c(P[("R16", s)], P[("CT1", s)], synapses=3)
        c(P[("CT1", s)], P[("LC4", s)], synapses=500)
        c(P[("CT1", s)], P[("LC10", s)], synapses=500)
        c(P[("CT1", s)], P[("LPLC2", s)], synapses=300)
        c(P[("LC4", s)], P[("GF", s)], synapses=60)
        c(P[("LPLC2", s)], P[("GF", s)], synapses=40)
        c(P[("LC10", s)], P[("DNa02", s)], synapses=60)
        c(P[("LC10", s)], P[("DNp09", s)], synapses=50)
        c(P[("HS", s)], P[("DNa01", s)], synapses=200)
        # hearing / wind / touch
        c(P[("JO_aud", s)], P[("AMMC_B", s)], synapses=16)
        c(P[("AMMC_B", s)], P[("pC1", s)], synapses=110)
        c(P[("AMMC_B", s)], P[("WED", s)], synapses=110)
        c(P[("WED", s)], P[("DNp09", s)], synapses=150)
        c(P[("JO_wind", s)], P[("AMMC_A", s)], synapses=16)
        c(P[("AMMC_A", s)], P[("DNa01", o)], synapses=110)
        c(P[("bristle", s)], P[("groomIN", s)], synapses=20)
        c(P[("groomIN", s)], P[("groomDN", s)], synapses=110)
        # heat / humidity
        c(P[("hot", s)], P[("hotPN", s)], synapses=40)
        c(P[("hotPN", s)], P[("LAL_av", s)], synapses=220)
        c(P[("moist", s)], P[("moistPN", s)], synapses=40)
        c(P[("moistPN", s)], P[("LAL_app", s)], synapses=150)
        # background brain: listens to many pathways, so activity spreads
        for src in ("LH_att", "LH_av", "G2N", "LC4", "AMMC_B", "MBON12", "LAL_app", "PN_DM1", "PN_DA2", "L2"):
            c(P[(src, s)], bg, p=0.06 if src != "L2" else 0.01, synapses=(40, 90))
        c(bg, P[("DNp09", s)], p=0.02, synapses=(2, 6))
    c(bg, bg, p=0.02, synapses=(10, 25))

    brain = b.build()
    return brain


__all__ = ["make_toy_brain", "GLOMERULI", "ATTRACTIVE", "AVERSIVE"]
