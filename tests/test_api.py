import pytest
from fastapi.testclient import TestClient

from api.routes import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint():
    """Test query endpoint returns expected structure."""
    response = client.post(
        "/query",
        json={"query": "test query"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "request_id" in data
    assert "latency_ms" in data


def test_query_with_request_id():
    """Test query endpoint respects X-Request-ID header."""
    test_id = "test-request-123"
    response = client.post(
        "/query",
        json={"query": "test query"},
        headers={"X-Request-ID": test_id}
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == test_id


def test_query_returns_grounded_fixture_answer():
    response = client.post(
        "/query",
        json={"query": "What are the public counter hours for a permit application?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["citations"]
    citation = data["citations"][0]
    assert {"document", "source", "chunk", "excerpt"} <= citation.keys()


def test_query_abstains_without_evidence():
    response = client.post("/query", json={"query": "What is the speed limit on the moon?"})
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is True
    assert data["grounded"] is False
    assert data["citations"] == []


def test_rebuild_requires_admin_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("WGLR_ADMIN_TOKEN", raising=False)
    response = client.post("/rebuild-index")
    assert response.status_code == 503

    monkeypatch.setenv("WGLR_ADMIN_TOKEN", "test-only-token")
    assert client.post("/rebuild-index", headers={"X-Admin-Token": "wrong"}).status_code == 401
    response = client.post("/rebuild-index", headers={"X-Admin-Token": "test-only-token"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
