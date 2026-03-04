"""Shared test fixtures for LegacyLens test suite."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from app.main import app
    return TestClient(app)
