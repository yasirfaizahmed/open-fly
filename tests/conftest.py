import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_home(tmp_path_factory):
    """Keep test downloads/caches out of the user's ~/.openfly (unless a real home is requested)."""
    if os.environ.get("OPENFLY_TEST_FLYWIRE") != "1":
        os.environ["OPENFLY_HOME"] = str(tmp_path_factory.mktemp("openfly_home"))
    os.environ.setdefault("OPENFLY_QUIET", "1")
    yield


@pytest.fixture(scope="session")
def toy():
    import openfly

    return openfly.load("toy")
