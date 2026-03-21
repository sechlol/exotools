"""Secret management for exotools using Pydantic settings and .env files."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExotoolsSecrets(BaseSettings):
    """Credentials expected in a project .env file (see `.env.example`)."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mast_token: str
    gaia_user: str
    gaia_password: str
    casjob_wsid: str
    casjob_password: str


def load_secrets(env_file: Path) -> ExotoolsSecrets:
    """Load secrets from ``env_file`` and the process environment."""
    return ExotoolsSecrets(_env_file=env_file)
