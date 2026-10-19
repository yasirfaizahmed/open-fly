"""``openfly`` command-line interface.

    openfly datasets                         list known brains
    openfly download flywire_783             fetch raw files and build the brain file
    openfly info flywire_783                 summary of a brain
    openfly playground [--dataset toy]       open the interactive web playground
    openfly simulate --scenario sugar_drop   run an experiment and print the results
    openfly path sugar_GRN MN9               strongest synaptic path between two groups
    openfly fly --arena banana               closed-loop virtual fly
    openfly learn --reinforcer reward        olfactory conditioning
    openfly screen MN9                       which cell types drive a target?
    openfly scenarios | odors | lessons      list what's available

Use ``--transmitters curated`` with FlyWire to apply literature-based
neurotransmitter fixes (see docs/05-simulation.md).
"""

from __future__ import annotations

import argparse
import sys


def _load(args):
    from .data import load

    return load(args.dataset, transmitters=getattr(args, "transmitters", None))


def cmd_datasets(args):
    from .data import datasets

    print(datasets().to_string())


def cmd_download(args):
    from .data import load

    b = load(args.dataset, rebuild=args.rebuild)
    print(b.summary())
    print(f"\nsaved: {b.meta.get('file', '(not cached)')}")


def cmd_info(args):
    b = _load(args)
    print(b.summary())
    if args.types:
        print("\nmost common cell types:")
        print(b.neurons["cell_type"].replace("", "?").value_counts().head(args.types).to_string())


def cmd_playground(args):
    from .playground import serve

    serve(args.dataset, host=args.host, port=args.port, open_browser=not args.no_browser, transmitters=args.transmitters)


def cmd_simulate(args):
    from . import motor, senses

    b = _load(args)
    sim = b.simulator(seed=args.seed)
    if args.scenario:
        sim.add(senses.scenario(b, args.scenario))
    if args.smell:
        sim.add(senses.smell(b, args.smell, args.intensity))
    if args.taste:
        sim.add(senses.taste(b, args.taste, args.intensity))
    for name in args.stimulate or []:
        sim.stimulate(name, args.rate)
    for name in args.silence or []:
        sim.silence(name)
    if not sim.stimuli:
        print("nothing to simulate: give --scenario, --smell, --taste or --stimulate", file=sys.stderr)
        return 2
    res = sim.run(args.duration, n_trials=args.trials, n_jobs=args.jobs, progress=args.trials > 1)
    print(res)
    print(f"(wall time {res.wall_time_s:.1f} s)\n")
    print("most active neurons (excluding stimulated):")
    print(res.top(args.top, exclude_driven=True).to_string())
    print("\nbehaviour:")
    print(motor.describe(res.behavior()))
    if args.movie:
        print(f"\nmovie: {res.movie(args.movie)}")
    if args.save:
        print(f"spikes: {res.save(args.save)}")


def cmd_path(args):
    from .analysis import strongest_path

    b = _load(args)
    df = strongest_path(b, args.source, args.target, min_fraction=args.min_fraction, max_paths=args.paths)
    print(df[["path", "step", "cell_type", "side", "nt", "synapses", "input_fraction"]].to_string(index=False))


def cmd_fly(args):
    from .agent import VirtualFly

    b = _load(args)
    fly = VirtualFly(b, args.arena, seed=args.seed, turn_highpass_s=3.0 if b.N > 20000 else None)
    fly.run(args.seconds, progress=True)
    print("\nevents:")
    for e in fly.events:
        print(f"  {e['t_s']:6.2f} s  {e['event']}")
    print(f"energy gathered: {fly.energy:.2f}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        ax = fly.plot()
        ax.figure.savefig(args.plot, dpi=110, bbox_inches="tight")
        print(f"plot: {args.plot}")


def cmd_learn(args):
    from .learning import OlfactoryConditioning

    b = _load(args)
    exp = OlfactoryConditioning(b, cs_plus=args.cs_plus, cs_minus=args.cs_minus, reinforcer=args.reinforcer)
    res = exp.run(n_trials=args.trials, odor_ms=args.odor_ms, test_ms=args.test_ms, progress=True)
    print(res.summary())


def cmd_screen(args):
    from .learning import activation_screen

    b = _load(args)
    print(activation_screen(b, args.target, top=args.top, model=args.model, progress=True).to_string())


def cmd_list(args):
    from . import senses
    from .playground import lessons

    if args.what == "scenarios":
        for s in senses.SCENARIOS.values():
            print(f"{s.name:<26} {s.title}")
    elif args.what == "odors":
        for o in senses.ODORS.values():
            print(f"{o.name:<22} {o.valence:<11} {', '.join(o.glomeruli)}")
    elif args.what == "tastes":
        print("\n".join(senses.list_tastants()))
    elif args.what == "lessons":
        for L in lessons.LESSONS:
            print(f"{L['title']}  ({L['minutes']} min)")
        for c in lessons.CHALLENGES:
            print(f"challenge: {c['title']}: {c['goal']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="openfly", description="Explore, simulate and train the fruit fly brain.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def brain_args(sp, default="toy"):
        sp.add_argument("--dataset", "-d", default=default, help=f"dataset name or .flybrain file (default {default})")
        sp.add_argument("--transmitters", choices=["shiu2024", "curated"], default=None,
                        help="neurotransmitter sign policy (default: as published)")

    sp = sub.add_parser("datasets", help="list known brains")
    sp.set_defaults(fn=cmd_datasets)

    sp = sub.add_parser("download", help="download and build a brain")
    sp.add_argument("dataset", nargs="?", default="flywire_783")
    sp.add_argument("--rebuild", action="store_true")
    sp.set_defaults(fn=cmd_download)

    sp = sub.add_parser("info", help="summary of a brain")
    sp.add_argument("dataset", nargs="?", default="toy")
    sp.add_argument("--transmitters", choices=["shiu2024", "curated"], default=None)
    sp.add_argument("--types", type=int, default=0, help="also list the N most common cell types")
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("playground", help="interactive web playground")
    brain_args(sp)
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(fn=cmd_playground)

    sp = sub.add_parser("simulate", help="run an experiment")
    brain_args(sp)
    sp.add_argument("--scenario")
    sp.add_argument("--smell")
    sp.add_argument("--taste")
    sp.add_argument("--intensity", type=float, default=1.0)
    sp.add_argument("--stimulate", action="append", help="neuron name to drive (repeatable)")
    sp.add_argument("--rate", type=float, default=150.0)
    sp.add_argument("--silence", action="append", help="neuron name to silence (repeatable)")
    sp.add_argument("--duration", type=float, default=500.0, help="ms")
    sp.add_argument("--trials", type=int, default=1)
    sp.add_argument("--jobs", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--movie", help="write a 3D HTML replay here")
    sp.add_argument("--save", help="save spikes (parquet) here")
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("path", help="strongest synaptic path")
    brain_args(sp)
    sp.add_argument("source")
    sp.add_argument("target")
    sp.add_argument("--min-fraction", type=float, default=0.01)
    sp.add_argument("--paths", type=int, default=1)
    sp.set_defaults(fn=cmd_path)

    sp = sub.add_parser("fly", help="closed-loop virtual fly")
    brain_args(sp)
    sp.add_argument("--arena", default="banana")
    sp.add_argument("--seconds", type=float, default=20.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--plot", help="save a trajectory plot (png)")
    sp.set_defaults(fn=cmd_fly)

    sp = sub.add_parser("learn", help="olfactory conditioning")
    brain_args(sp)
    sp.add_argument("--cs-plus", default="banana")
    sp.add_argument("--cs-minus", default="acetic_acid")
    sp.add_argument("--reinforcer", default="reward", choices=["reward", "punishment", "sugar", "bitter"])
    sp.add_argument("--trials", type=int, default=3)
    sp.add_argument("--odor-ms", type=float, default=800.0)
    sp.add_argument("--test-ms", type=float, default=500.0)
    sp.set_defaults(fn=cmd_learn)

    sp = sub.add_parser("screen", help="activation screen: who drives a target?")
    brain_args(sp)
    sp.add_argument("target")
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--model", choices=["rate", "lif"], default="rate")
    sp.set_defaults(fn=cmd_screen)

    for what in ("scenarios", "odors", "tastes", "lessons"):
        sp = sub.add_parser(what, help=f"list {what}")
        sp.set_defaults(fn=cmd_list, what=what)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
