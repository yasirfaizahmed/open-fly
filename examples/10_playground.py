"""Start the playground from Python (e.g. a notebook) and drive it by HTTP."""

import json
import time
import urllib.request

from openfly.playground import start

pg = start("toy", port=8765)
print("open", pg.url)


def command(c):
    req = urllib.request.Request(pg.url + "api/command", data=json.dumps(c).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


print(command({"cmd": "scenario", "name": "ripe_banana", "replace": True}))
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pg.stop()
