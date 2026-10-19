"""Curated names for famous fly neurons and populations.

Connectome tables use systematic names (``CB0701``, ``DNp01``...). Biology
papers use memorable ones ("MN9", "the giant fiber", "sugar neurons").
This module connects the two, so you can write ``brain["MN9"]`` or
``brain["giant_fiber"]``.

Every alias is a *selector*, a set of column filters, plus a short
description and the paper it comes from. The selectors use FlyWire
(FAFB v783) annotation names; on other brains an alias simply finds
nothing if those names do not exist there.

Filter values:

* ``"text"`` matches a column exactly,
* ``["a", "b"]`` matches any of several values,
* ``"re:^PAM"`` is a regular expression.

A selector can also be a *list* of such dicts: alternatives tried in
order, for datasets that name things differently (FlyWire ``DNp01`` is
the hemibrain's ``Giant Fiber``).

Add your own with :func:`register_alias`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Alias:
    name: str
    selector: dict | list  # a dict of column filters, or a list of alternatives
    description: str
    reference: str = ""
    group: str = "misc"
    confidence: str = "established"  # established | likely | speculative
    tags: tuple = field(default_factory=tuple)


ALIASES: dict[str, Alias] = {}


def register_alias(
    name: str,
    selector: dict | list,
    description: str,
    reference: str = "",
    group: str = "misc",
    confidence: str = "established",
) -> Alias:
    """Add (or replace) an alias. Names are case-insensitive."""
    sel = [dict(x) for x in selector] if isinstance(selector, list) else dict(selector)
    a = Alias(name, sel, description, reference, group, confidence)
    ALIASES[name.lower()] = a
    return a


def get_alias(name: str) -> Alias | None:
    return ALIASES.get(str(name).lower())


def _r(name, selector, description, reference="", group="misc", confidence="established"):
    register_alias(name, selector, description, reference, group, confidence)


# ------------------------------------------------------------------ taste
_r("sugar_GRN", {"cell_class": "gustatory", "cell_sub_class": ["sugar", "sugar/low_salt"]},
   "Labellar gustatory receptor neurons that respond to sugar. Activating them drives proboscis extension.",
   "Shiu et al. 2024 Nature; Tastekin et al. 2026 Cell (modality typing)", "taste")
_r("bitter_GRN", {"cell_class": "gustatory", "cell_sub_class": "bitter"},
   "Labellar GRNs for bitter compounds. They suppress feeding.", "Tastekin et al. 2026 Cell", "taste")
_r("water_GRN", {"cell_class": "gustatory", "cell_sub_class": "water"},
   "Labellar GRNs that respond to water (low osmolarity).", "Tastekin et al. 2026 Cell", "taste")
_r("salt_high_GRN", {"cell_class": "gustatory", "cell_sub_class": "high_salt/heavy_metal"},
   "GRNs for high salt and heavy metals (aversive).", "Tastekin et al. 2026 Cell", "taste")
_r("glutamate_GRN", {"cell_class": "gustatory", "cell_sub_class": "glutamate"},
   "GRNs putatively tuned to glutamate (savoury/amino-acid taste).", "Tastekin et al. 2026 Cell", "taste", "likely")
_r("taste_peg_GRN", {"cell_class": "gustatory", "cell_sub_class": "taste peg"},
   "GRNs of the taste pegs inside the labellum (contact with food during feeding).", "Tastekin et al. 2026 Cell", "taste")
_r("pharyngeal_GRN", {"cell_class": "gustatory", "cell_type": "re:^PhG"},
   "Pharyngeal taste neurons that check food after it enters the mouth.", "Tastekin et al. 2026 Cell", "taste")

# ------------------------------------------------------------- olfaction
_r("ORN", {"cell_class": "olfactory"}, "All olfactory receptor neurons (antenna + maxillary palp).",
   "Schlegel et al. 2024 Nature", "smell")
_r("ALPN", {"cell_class": "ALPN"}, "Antennal lobe projection neurons: carry odour identity to mushroom body and lateral horn.",
   "Schlegel et al. 2024 Nature", "smell")
_r("uPN", {"cell_class": "ALPN", "cell_sub_class": "uniglomerular"}, "Uniglomerular projection neurons (one glomerulus each).",
   "Schlegel et al. 2024 Nature", "smell")
_r("LHN", {"cell_class": ["LHLN", "LHCENT"]}, "Lateral horn local / centrifugal neurons (innate odour responses).",
   "Schlegel et al. 2024 Nature", "smell")
_r("CO2_ORN", {"cell_type": "ORN_V"}, "Gr21a/Gr63a neurons that sense CO2 (an alarm signal for flies).",
   "Suh et al. 2004 Nature", "smell")
_r("geosmin_ORN", {"cell_type": "ORN_DA2"}, "Or56a neurons, tuned to geosmin (smell of harmful moulds).",
   "Stensmyr et al. 2012 Cell", "smell")
_r("cVA_ORN", {"cell_type": "ORN_DA1"}, "Or67d neurons, tuned to the male pheromone cVA.", "Kurtovic et al. 2007 Nature", "smell")
_r("vinegar_ORN", {"cell_type": ["ORN_DM1", "ORN_VA2"]}, "Or42b/Or92a neurons, the attractive vinegar channels.",
   "Semmelhack & Wang 2009 Nature", "smell")

# ------------------------------------------------------- mushroom body
_r("KC", [{"cell_class": "Kenyon_Cell"}, {"cell_type": "re:^KC"}], "Kenyon cells: ~5,000 sparse odour coders of the mushroom body (learning centre).",
   "Aso et al. 2014 eLife; Li et al. 2020 eLife", "learning")
_r("MBON", [{"cell_class": "MBON"}, {"cell_type": "re:^MBON"}], "Mushroom body output neurons: read out learned valence.", "Aso et al. 2014 eLife", "learning")
_r("DAN", [{"cell_class": "DAN"}, {"cell_type": "re:^(PAM|PPL)"}], "Dopaminergic neurons of the mushroom body (teaching signals).", "Aso et al. 2014 eLife", "learning")
_r("PAM", {"cell_type": "re:^PAM"}, "PAM dopaminergic cluster: mostly reward signals.", "Liu et al. 2012 Nature; Burke et al. 2012 Nature", "learning")
_r("PPL1", {"cell_type": "re:^PPL1"}, "PPL1 dopaminergic cluster: mostly punishment signals.", "Claridge-Chang et al. 2009 Cell", "learning")
_r("APL", {"cell_type": "APL"}, "Anterior paired lateral neuron: one giant GABAergic neuron that keeps KC activity sparse.",
   "Lin et al. 2014 Nat Neurosci", "learning")
_r("DPM", {"cell_type": "DPM"}, "Dorsal paired medial neuron (memory consolidation).", "Waddell et al. 2000 Cell", "learning")

# ---------------------------------------------------------------- vision
_r("photoreceptors", {"cell_type": ["R1-6", "R7", "R8"]}, "All photoreceptors of the compound eyes.",
   "Matsliah et al. 2024 Nature", "vision")
_r("R1-6", {"cell_type": "R1-6"}, "Outer photoreceptors: broadband luminance and motion.", "Matsliah et al. 2024 Nature", "vision")
_r("R7", {"cell_type": "R7"}, "Inner photoreceptor R7 (UV).", "Matsliah et al. 2024 Nature", "vision")
_r("R8", {"cell_type": "R8"}, "Inner photoreceptor R8 (blue/green).", "Matsliah et al. 2024 Nature", "vision")
_r("ocellar_PR", {"cell_class": "visual", "cell_sub_class": "ocellar"}, "Photoreceptors of the three ocelli (horizon/attitude).",
   "Schlegel et al. 2024 Nature", "vision")
_r("T4", {"cell_type": ["T4a", "T4b", "T4c", "T4d"]}, "ON motion detectors, one subtype per cardinal direction.",
   "Maisak et al. 2013 Nature", "vision")
_r("T5", {"cell_type": ["T5a", "T5b", "T5c", "T5d"]}, "OFF motion detectors, one subtype per cardinal direction.",
   "Maisak et al. 2013 Nature", "vision")
_r("LC4", {"cell_type": "LC4"}, "Lobula columnar 4: looming detector that excites the giant fiber.", "von Reyn et al. 2017 Neuron", "vision")
_r("LPLC2", {"cell_type": "LPLC2"}, "Looming-sensitive visual projection neurons.", "Klapoetke et al. 2017 Nature", "vision")
_r("LC10", {"cell_type": "re:^LC10"}, "Small-object detectors used to chase a mate.", "Ribeiro et al. 2018 Cell", "vision")
_r("LC11", {"cell_type": "LC11"}, "Small moving-object detector.", "Keleş & Frye 2017 Curr Biol", "vision")
_r("HS", {"cell_type": ["HSE", "HSN", "HSS"]}, "Horizontal system cells: wide-field yaw motion.", "Hausen 1982", "vision")
_r("VS", {"cell_type": "re:^VS"}, "Vertical system cells: wide-field pitch/roll motion.", "Hengstenberg 1982", "vision")

# --------------------------------------------------------- mechanosense
_r("JO_auditory", {"cell_class": "mechanosensory", "cell_sub_class": "auditory"},
   "Johnston's organ neurons for sound, e.g. courtship song (JO-A/B).", "Kamikouchi et al. 2009 Nature", "hearing")
_r("JO_wind_gravity", {"cell_class": "mechanosensory", "cell_sub_class": "wind_gravity"},
   "Johnston's organ neurons for wind and gravity (JO-C/E).", "Yorozu et al. 2009 Nature", "hearing")
_r("head_bristle", {"cell_class": "mechanosensory", "cell_sub_class": "head bristle"},
   "Touch-sensitive bristle neurons on the head (trigger grooming).", "Hampel et al. 2017 eLife", "touch")
_r("eye_bristle", {"cell_class": "mechanosensory", "cell_sub_class": "eye bristle"},
   "Interommatidial bristle neurons on the eye (trigger eye grooming).", "Hampel et al. 2017 eLife", "touch")

# ---------------------------------------------------------- thermo/hygro
_r("hot_TRN", {"cell_type": "TRN_VP2"}, "Hot-sensing thermoreceptor neurons of the arista (VP2).", "Frank et al. 2015 Nature", "temperature")
_r("cold_TRN", {"cell_type": ["TRN_VP3a", "TRN_VP3b", "TRN_VP1m"]}, "Cold-sensing thermoreceptor neurons (VP3, VP1m).",
   "Frank et al. 2015 Nature; Marin et al. 2020 Curr Biol", "temperature", "likely")
_r("dry_HRN", {"cell_type": "HRN_VP4"}, "Dry-sensing hygroreceptor neurons (VP4).", "Knecht et al. 2017 eLife", "humidity", "likely")
_r("moist_HRN", {"cell_type": ["HRN_VP1d", "HRN_VP1l", "HRN_VP5"]}, "Moist/other hygroreceptor neurons (VP1, VP5).",
   "Marin et al. 2020 Curr Biol", "humidity", "likely")

# ------------------------------------------------------------- central
_r("EPG", {"cell_type": ["EPG", "EPGt"]}, "Compass neurons of the ellipsoid body (heading).", "Seelig & Jayaraman 2015 Nature", "navigation")
_r("PFL3", {"cell_type": "PFL3"}, "Central-complex steering output neurons.", "Westeinde et al. 2024 Nature", "navigation")
_r("steering_premotor",
   {"super_class": ["descending", "central"],
    "neuropil": ["LAL_L", "LAL_R", "SPS_L", "SPS_R", "IPS_L", "IPS_R", "LAL", "SPS", "IPS"]},
   "Descending and central neurons whose main neuropil is a premotor steering region (LAL, SPS, IPS).",
   "Namiki et al. 2018 eLife; Rayshubskiy et al. 2020", "navigation", "likely")
_r("pC1", {"cell_type": "re:^pC1"}, "Female social-behaviour neurons (receptivity, aggression).", "Wang et al. 2021 Neuron", "social")

# ------------------------------------------------------- motor / output
_r("MN9", {"cell_type": "CB0701"}, "Motor neuron 9: drives the proboscis rostrum (feeding). The model's go-to feeding readout.",
   "Shiu et al. 2024 Nature; Gordon & Scott 2009 Neuron", "motor")
_r("proboscis_MN", {"cell_sub_class": "proboscis_motor_neuron"}, "Motor neurons of the proboscis.", "Schlegel et al. 2024 Nature", "motor")
_r("ingestion_MN", {"cell_sub_class": "ingestion_motor_neuron"}, "Motor neurons for pumping/ingestion.", "Schlegel et al. 2024 Nature", "motor")
_r("antennal_MN", {"cell_sub_class": "antennal_motor_neuron"}, "Motor neurons that move the antennae.", "Schlegel et al. 2024 Nature", "motor")
_r("neck_MN", {"cell_sub_class": "neck_motor_neuron"}, "Neck motor neurons (head turning).", "Schlegel et al. 2024 Nature", "motor")
_r("descending", {"super_class": "descending"}, "All descending neurons: the brain's ~1,300 output wires to the body.",
   "Schlegel et al. 2024 Nature", "motor")
_r("giant_fiber", [{"cell_type": "DNp01"}, {"cell_type": "Giant Fiber"}], "The giant fiber (DNp01): command neuron for the escape jump.", "Allen et al. 2006; von Reyn et al. 2014", "motor")
_r("GF", [{"cell_type": "DNp01"}, {"cell_type": "Giant Fiber"}], "Alias of giant_fiber.", "von Reyn et al. 2014 Nat Neurosci", "motor")
_r("MDN", {"cell_type": "MDN"}, "Moonwalker descending neurons: command backward walking.", "Bidaye et al. 2014 Science", "motor")
_r("DNa02", {"cell_type": "DNa02"}, "Steering DN: activity on one side turns the fly toward that side.", "Rayshubskiy et al. 2020 bioRxiv; Yang et al. 2024 Cell", "motor")
_r("DNa01", {"cell_type": "DNa01"}, "Steering DN (slower, sustained turns).", "Rayshubskiy et al. 2020 bioRxiv", "motor")
_r("P9", {"cell_type": "DNp09"}, "DNp09, thought to correspond to the P9 forward-walking/steering command neurons.",
   "Bidaye et al. 2020 Neuron; Braun et al. 2024 Nature", "motor", "likely")
_r("DNp09", {"cell_type": "DNp09"}, "Forward walking / object-directed steering DN.", "Braun et al. 2024 Nature", "motor", "likely")
_r("oviDN", {"cell_type": "re:^oviDN"}, "Oviposition descending neurons (egg laying).", "Wang et al. 2020 Neuron", "motor", "likely")


def all_aliases() -> list[Alias]:
    return list(ALIASES.values())
