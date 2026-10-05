import os
import json
import pytest
import ee


@pytest.fixture(scope="session")
def ee_session():
    """Initialize Earth Engine once per test session.

    In CI (GitHub Actions): authenticate via a service account whose
    JSON key is stored in the GEE_SERVICE_ACCOUNT_KEY secret.
    Locally: use the interactive persistent credentials from
    `earthengine authenticate`.
    """
    import json as _json, os as _os, pytest as _pytest
    if _os.environ.get('GITHUB_ACTIONS') == 'true':
        _key = _os.environ.get('GEE_SERVICE_ACCOUNT_KEY', '')
        if not _key.strip():
            if _os.environ.get('EE_OPTIONAL') == 'true':
                _pytest.skip('no Earth Engine credentials in this run (a fork or Dependabot pull request has no Actions secrets)')
            raise RuntimeError('GEE_SERVICE_ACCOUNT_KEY is empty on a trusted GitHub Actions run: check the repository secret')
        try:
            _json.loads(_key)
        except ValueError:
            raise ValueError('GEE_SERVICE_ACCOUNT_KEY is set but is not valid JSON (%d characters)' % len(_key)) from None
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
