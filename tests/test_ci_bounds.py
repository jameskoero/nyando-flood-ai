"""CI never waits without a limit: the test job and step have time limits, the GDAL network reads in the GFM tests have timeouts and retries, and a test that runs over five minutes prints its stack."""
import re
from pathlib import Path
import pytest

CI = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def ee_session():
    """Override tests/conftest.py: this check never touches Earth Engine."""
    yield


def test_the_job_and_the_test_step_have_time_limits():
    job, step = re.search(r"^    timeout-minutes: (\d+)$", CI, re.M), re.search(r"- name: Run tests\n        timeout-minutes: (\d+)\n", CI)
    assert job and step and int(step.group(1)) < int(job.group(1)) <= 60


def test_remote_raster_reads_are_bounded():
    for key, bound in (("GDAL_HTTP_CONNECTTIMEOUT", 60), ("GDAL_HTTP_TIMEOUT", 300), ("GDAL_HTTP_MAX_RETRY", 5), ("GDAL_HTTP_RETRY_DELAY", 30)):
        m = re.search(r"^          %s: \"(\d+)\"$" % key, CI, re.M)
        assert m and 0 < int(m.group(1)) <= bound, key


def test_a_slow_test_prints_its_stack():
    assert re.search(r"-o faulthandler_timeout=\d+", CI)
# END OF FILE
