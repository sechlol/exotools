import os
from pathlib import Path

_CURRENT_DIR = Path(os.path.realpath(__file__)).parent
_TEST_ASSETS_DIR = _CURRENT_DIR / "test_assets"

TEST_TMP_DIR = _CURRENT_DIR / "tmp"
TEST_ASSETS_QTABLES = _TEST_ASSETS_DIR / "qtables"
