import os
import json
import pytest
import ee


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Initialize Earth Engine once per test session.

    In CI (GitHub Actions): authenticate via a service account whose
    JSON key is stored in the GEE_SERVICE_ACCOUNT_KEY secret.
    Locally: use the interactive persistent credentials from
    `earthengine authenticate`.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        key_json = os.environ["GEE_SERVICE_ACCOUNT_KEY"]
        key_dict = json.loads(key_json)
        credentials = ee.ServiceAccountCredentials(
            email=key_dict["client_email"],
            key_data=key_json,
        )
        ee.Initialize(credentials)
    else:
        ee.Authenticate()
        ee.Initialize()

    yield
