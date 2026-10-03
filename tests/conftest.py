"""No automated backend test may reach a real explanation provider."""
import pytest


@pytest.fixture(autouse=True)
def forbid_live_api(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Live HTTPS calls are forbidden in backend tests')
    monkeypatch.setattr('http.client.HTTPSConnection.connect', forbidden)
    monkeypatch.setattr('http.client.HTTPSConnection.request', forbidden)
