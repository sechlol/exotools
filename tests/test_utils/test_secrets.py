"""Tests for the secrets utility module."""

from pathlib import Path

import pytest

from exotools.utils.secrets import ExotoolsSecrets, load_secrets

_SECRET_ENV_VARS = ("MAST_TOKEN", "GAIA_USER", "GAIA_PASSWORD", "CASJOB_WSID", "CASJOB_PASSWORD")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real environment variables from leaking into tests."""
    for var in _SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    """Create a temporary .env file with valid secrets."""
    env = tmp_path / ".env"
    env.write_text(
        "MAST_TOKEN=test-token\n"
        "GAIA_USER=test-user\n"
        "GAIA_PASSWORD=test-pass\n"
        "CASJOB_WSID=99999999\n"
        "CASJOB_PASSWORD=test-casjob-pass\n"
    )
    return env


class TestExotoolsSecrets:
    def test_loads_all_fields_from_env_file(self, env_file: Path):
        secrets = ExotoolsSecrets(_env_file=env_file)

        assert secrets.mast_token == "test-token"
        assert secrets.gaia_user == "test-user"
        assert secrets.gaia_password == "test-pass"
        assert secrets.casjob_wsid == "99999999"
        assert secrets.casjob_password == "test-casjob-pass"

    def test_extra_vars_are_ignored(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text(
            "MAST_TOKEN=t\n"
            "GAIA_USER=u\n"
            "GAIA_PASSWORD=p\n"
            "CASJOB_WSID=1\n"
            "CASJOB_PASSWORD=p\n"
            "SOME_OTHER_VAR=should-be-ignored\n"
        )
        secrets = ExotoolsSecrets(_env_file=env)
        assert not hasattr(secrets, "some_other_var")

    def test_missing_field_raises(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("MAST_TOKEN=t\n")

        with pytest.raises(Exception):
            ExotoolsSecrets(_env_file=env)

    def test_env_vars_override_file(self, env_file: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MAST_TOKEN", "from-env")
        secrets = ExotoolsSecrets(_env_file=env_file)
        assert secrets.mast_token == "from-env"


class TestLoadSecrets:
    def test_returns_secrets_instance(self, env_file: Path):
        secrets = load_secrets(env_file)

        assert isinstance(secrets, ExotoolsSecrets)
        assert secrets.mast_token == "test-token"
        assert secrets.gaia_user == "test-user"

    def test_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(Exception):
            load_secrets(tmp_path / "nonexistent.env")
