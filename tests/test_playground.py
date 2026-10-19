"""The playground server end to end (HTTP + SSE), on the toy brain."""

import json
import urllib.request

import pytest

from openfly.playground import start
from openfly.playground.lessons import CHALLENGES, LESSONS, check_challenge


@pytest.fixture(scope="module")
def pg(toy):
    p = start(toy, port=0)
    yield p
    p.stop()


def get(pg, path):
    return urllib.request.urlopen(pg.url + path, timeout=10).read()


def cmd(pg, c):
    req = urllib.request.Request(pg.url + "api/command", data=json.dumps(c).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def frames(pg, n):
    r = urllib.request.urlopen(pg.url + "api/stream", timeout=10)
    out = []
    while len(out) < n:
        line = r.readline()
        if line.startswith(b"data:"):
            out.append(json.loads(line[5:]))
    r.close()
    return out


def test_static_and_info(pg, toy):
    assert b"openfly" in get(pg, "")
    assert b"BrainViewer" in get(pg, "static/js/viewer.js")
    info = json.loads(get(pg, "api/info"))
    assert info["N"] == toy.N and info["scenarios"] and info["lessons"]
    geo = get(pg, "api/geometry")
    assert len(geo) == toy.N * 15


def test_stimulus_and_stream(pg):
    r = cmd(pg, {"cmd": "scenario", "name": "sugar_drop", "replace": True})
    assert r["ok"] and r["n"] == 16
    fs = frames(pg, 12)
    assert fs[-1]["stimuli"] and fs[-1]["drive"]["feeding"] > 0.3
    assert fs[-1]["t"] > fs[0]["t"]
    assert cmd(pg, {"cmd": "clear"})["ok"]


def test_commands(pg):
    assert cmd(pg, {"cmd": "path", "source": "sugar_GRN", "target": "MN9"})["path"][-1]["cell_type"] == "CB0701"
    assert cmd(pg, {"cmd": "neuron", "index": 0})["neuron"]["cell_type"]
    assert cmd(pg, {"cmd": "search", "q": "giant"})["results"]
    assert cmd(pg, {"cmd": "silence", "name": "G2N-1"})["n"] == 6
    assert cmd(pg, {"cmd": "unsilence"})["ok"]
    assert cmd(pg, {"cmd": "influence", "source": "bitter_GRN"})["values"]
    assert not cmd(pg, {"cmd": "nope"})["ok"]
    assert "Did you mean" in cmd(pg, {"cmd": "stimulate", "name": "MN99"})["error"]
    for c in ({"cmd": "smell", "odor": "banana", "left": 1, "right": 0.2}, {"cmd": "taste", "tastant": "bitter"},
              {"cmd": "vision", "kind": "looming", "azimuth": -40}, {"cmd": "hear"}, {"cmd": "wind"},
              {"cmd": "touch"}, {"cmd": "temperature", "kind": "hot"}, {"cmd": "stimulate", "name": "MDN"}):
        assert cmd(pg, c)["ok"], c


def test_fly_mode(pg):
    r = cmd(pg, {"cmd": "mode", "mode": "fly", "arena": "banana"})
    assert r["ok"] and r["arena"]["items"] and r["field"]["n"] == 48
    fs = frames(pg, 5)
    assert fs[-1]["fly"] is not None and fs[-1]["mode"] == "fly"
    assert cmd(pg, {"cmd": "arena_add", "kind": "heat", "x": 10, "y": 10})["ok"]
    assert cmd(pg, {"cmd": "fly_place", "x": 30, "y": 30})["ok"]
    assert cmd(pg, {"cmd": "mode", "mode": "brain"})["ok"]


def test_jobs(pg):
    import time

    j = cmd(pg, {"cmd": "job", "kind": "screen", "target": "MN9"})["job"]
    for _ in range(60):
        st = json.loads(get(pg, f"api/job/{j}"))
        if st["status"] != "running":
            break
        time.sleep(0.5)
    assert st["status"] == "done" and st["result"]["table"][0]["group"] == "G2N-1"


def test_lessons_and_challenges():
    assert all(L["steps"] for L in LESSONS)
    ids = {c["id"] for c in CHALLENGES}
    assert check_challenge("moonwalk", {"drive": {"walk_backward": 0.6}})["ok"]
    assert not check_challenge("puppet", {"drive": {"feeding": 0.9}, "kinds": ["taste:sugar"]})["ok"]
    assert check_challenge("sharpshooter", {"drive": {"feeding": 0.5}, "n_driven": 6})["ok"]
    assert "moonwalk" in ids
