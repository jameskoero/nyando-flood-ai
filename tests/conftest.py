"""
Shared pytest configuration for this test suite.

Earth Engine requires ee.Initialize() to run inside the SAME process that
executes ee.* calls. A notebook cell's ee.Initialize() does not propagate
to a `!pytest` subprocess — this fixture initializes EE once, in the
pytest process itself, for every test that needs it.

Requires ee.Authenticate() to have been run at least once in this
container (persists a credential to disk); this fixture does not
perform interactive auth, only ee.Initialize() against that credential.
"""

import pytest
import ee

GEE_PROJECT_ID = "nyando-flood-ai"


@pytest.fixture(scope="session", autouse=True)
def _initialize_earth_engine():
    ee.Initialize(project=GEE_PROJECT_ID)
