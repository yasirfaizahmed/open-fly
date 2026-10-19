"""Guided lessons and challenges for the playground.

A lesson is a short sequence of steps. Each step explains one idea and
can set up the brain for you (``action``), so you just watch. Challenges
are puzzles ("make the fly walk backwards without touching it") that the
server checks against the live simulation.

Everything here is plain data, so adding a lesson is just appending to
:data:`LESSONS`. Actions are the same JSON commands the UI sends to
``/api/command``.
"""

from __future__ import annotations

LESSONS = [
    {
        "id": "meet",
        "title": "1. Meet the brain",
        "minutes": 3,
        "steps": [
            {"title": "139,000 points of light",
             "text": "Every dot is one neuron, placed where FlyWire found it in the electron-microscopy volume "
                     "(or a schematic position for brains without coordinates). Drag to rotate, scroll to zoom, "
                     "right-drag to pan.",
             "action": {"cmd": "color", "mode": "class"}},
            {"title": "Two eyes, one central brain",
             "text": "Colour is the 'super class'. The two big lobes on the sides are the optic lobes, more than half "
                     "of all neurons, processing vision. The middle is the central brain. Sensory neurons enter from "
                     "the antennae, eyes and mouthparts.",
             "action": {"cmd": "highlight", "name": "photoreceptors"}},
            {"title": "Excitation and inhibition",
             "text": "Switch the colours to neurotransmitters. Acetylcholine (excitatory) dominates; GABA and "
                     "glutamate are inhibitory in the model; dopamine, serotonin and octopamine are neuromodulators.",
             "action": {"cmd": "color", "mode": "nt"}},
            {"title": "Click a neuron",
             "text": "Click any dot to see its cell type, side, transmitter and its strongest partners.",
             "action": {"cmd": "color", "mode": "class"}},
        ],
    },
    {
        "id": "taste",
        "title": "2. From taste to action",
        "minutes": 4,
        "steps": [
            {"title": "A drop of sugar",
             "text": "We drive the sugar-sensing gustatory receptor neurons. Watch the signal spread into the "
                     "subesophageal zone and the 'feeding' meter rise as MN9, the proboscis motor neuron, fires. "
                     "This is the central result of Shiu et al. (2024).",
             "action": {"cmd": "scenario", "name": "sugar_drop", "replace": True}},
            {"title": "Who carries the message?",
             "text": "Click 'Path' in the Lab tab with sugar_GRN -> MN9 to see the strongest synaptic chain.",
             "action": {"cmd": "path", "source": "sugar_GRN", "target": "MN9"}},
            {"title": "Add something bitter",
             "text": "Bitter neurons inhibit the feeding circuit. Compare the feeding meter with sugar alone.",
             "action": {"cmd": "scenario", "name": "sugar_laced_with_bitter", "replace": True}},
        ],
    },
    {
        "id": "smell",
        "title": "3. A smell's journey",
        "minutes": 4,
        "steps": [
            {"title": "Banana!",
             "text": "Fruity esters activate a handful of olfactory receptor neuron types, each converging on one "
                     "glomerulus of the antennal lobe.",
             "action": {"cmd": "scenario", "name": "ripe_banana", "replace": True}},
            {"title": "Sparse coding in the mushroom body",
             "text": "Projection neurons fan out to ~5,000 Kenyon cells, but only a few percent respond to any "
                     "odour. That sparse, odour-specific code is what makes learning possible.",
             "action": {"cmd": "highlight", "name": "KC"}},
            {"title": "A bad smell",
             "text": "Geosmin (mould) has a dedicated line, ORN_DA2 -> DA2 projection neurons. Innate aversion needs "
                     "no learning.",
             "action": {"cmd": "scenario", "name": "mouldy_fruit", "replace": True}},
        ],
    },
    {
        "id": "danger",
        "title": "4. Danger from above",
        "minutes": 3,
        "steps": [
            {"title": "A looming shadow",
             "text": "An expanding dark disk is the visual signature of a collision course. Looming detectors (LC4, "
                     "LPLC2) excite the giant fiber, the fastest escape command in the fly.",
             "action": {"cmd": "scenario", "name": "predator_from_right", "replace": True}},
            {"title": "Repeat it",
             "text": "Press the scenario again and watch the escape meter. Notice which side lights up first.",
             "action": None},
        ],
    },
    {
        "id": "lesion",
        "title": "5. Lesion studio",
        "minutes": 4,
        "steps": [
            {"title": "Necessary or not?",
             "text": "Biologists silence neurons to test whether they are needed. A virtual screen on FlyWire "
                     "suggests a few neuron types are essential for sugar-evoked feeding. Silence CB0553 and "
                     "taste sugar again.",
             "action": {"cmd": "silence", "name": "CB0553"}},
            {"title": "Taste sugar with CB0553 silenced",
             "text": "If the feeding meter stays flat, the model says CB0553 is necessary. That's a prediction a "
                     "wet lab could test.",
             "action": {"cmd": "scenario", "name": "sugar_drop", "replace": True}},
            {"title": "Restore",
             "text": "Clear silencing and compare.",
             "action": {"cmd": "unsilence"}},
        ],
    },
    {
        "id": "learning",
        "title": "6. Teaching a fly",
        "minutes": 6,
        "steps": [
            {"title": "Dopamine writes memories",
             "text": "In the Lab tab, run 'Olfactory conditioning'. An odour is paired with dopamine (reward or "
                     "punishment); synapses from the odour's Kenyon cells onto specific output neurons weaken; "
                     "afterwards the same odour drives the fly differently.",
             "action": {"cmd": "highlight", "name": "MBON"}},
        ],
    },
    {
        "id": "loop",
        "title": "7. Closing the loop",
        "minutes": 5,
        "steps": [
            {"title": "Give the brain a body",
             "text": "Switch to 'Virtual fly'. The brain now smells what the fly's antennae smell, tastes what it "
                     "stands on, sees food and predators, and its descending neurons move the body.",
             "action": {"cmd": "mode", "mode": "fly", "arena": "banana"}},
            {"title": "A predator",
             "text": "This arena has a predator that attacks after a few seconds. Watch for the escape jump.",
             "action": {"cmd": "mode", "mode": "fly", "arena": "predator"}},
        ],
    },
]

CHALLENGES = [
    {"id": "puppet", "title": "Puppet master",
     "goal": "Make the fly extend its proboscis (feeding > 0.5) without any taste stimulus.",
     "hint": "Search for neurons upstream of MN9, e.g. try the Lab's activation screen."},
    {"id": "moonwalk", "title": "Moonwalker",
     "goal": "Make the fly walk backwards (walk_backward > 0.4).",
     "hint": "MDN means 'moonwalker descending neuron'... but can you do it with a smell or a taste?"},
    {"id": "escape_artist", "title": "Escape artist",
     "goal": "Trigger the escape response (> 0.5) without any visual stimulus.",
     "hint": "The giant fiber also listens to the antennae."},
    {"id": "bitter_wins", "title": "Bitter beats sweet",
     "goal": "Keep sugar on but feeding below 0.1.",
     "hint": "Bitter and high-salt neurons recruit inhibition."},
    {"id": "sharpshooter", "title": "Neural sharpshooter",
     "goal": "Reach feeding > 0.3 while directly stimulating 10 neurons or fewer.",
     "hint": "Stimulate a single strong driver instead of a whole sensory population."},
]


def check_challenge(cid: str, state: dict) -> dict:
    """Evaluate a challenge. ``state`` has ``drive`` (behaviour), ``kinds`` (active stimulus kinds) and ``n_driven``."""
    d = state.get("drive", {})
    kinds = set(state.get("kinds", []))
    n = int(state.get("n_driven", 0))
    tastes = {"taste"} & kinds or {k for k in kinds if k.startswith("taste")}
    rules = {
        "puppet": (d.get("feeding", 0) > 0.5 and not tastes, f"feeding {d.get('feeding', 0):.2f}"),
        "moonwalk": (d.get("walk_backward", 0) > 0.4, f"walk_backward {d.get('walk_backward', 0):.2f}"),
        "escape_artist": (d.get("escape", 0) > 0.5 and not any(k.startswith("vision") for k in kinds),
                          f"escape {d.get('escape', 0):.2f}"),
        "bitter_wins": ("taste:sugar" in kinds and d.get("feeding", 1) < 0.1, f"feeding {d.get('feeding', 0):.2f}"),
        "sharpshooter": (d.get("feeding", 0) > 0.3 and 0 < n <= 10, f"feeding {d.get('feeding', 0):.2f} with {n} neurons"),
    }
    if cid not in rules:
        return {"id": cid, "ok": False, "detail": "unknown challenge"}
    ok, detail = rules[cid]
    return {"id": cid, "ok": bool(ok), "detail": detail}
