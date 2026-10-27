import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.synthetic import make_panel  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_panel():
    return make_panel()
