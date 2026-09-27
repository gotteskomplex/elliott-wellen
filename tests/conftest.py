"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from elliott.settings import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()
