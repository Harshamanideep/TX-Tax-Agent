"""API tests with the agent replaced by a fake, so they need no API key."""
from fastapi.testclient import TestClient

import src.api as api


def fake_run(question, thread_id):
    return {"answer": f"echo: {question}", "tools_used": ["search_tax_instructions(query='x')"]}


def make_client(monkeypatch, run=fake_run):
    monkeypatch.setattr(api, "run", run)
    monkeypatch.setattr(api, "get_agent", lambda: None)
    monkeypatch.setattr(api, "get_store", lambda: None)
    return TestClient(api.app)


def test_ask_returns_answer_and_new_thread(monkeypatch):
    with make_client(monkeypatch) as client:
        r = client.post("/ask", json={"question": "When is TC-40 due?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo: When is TC-40 due?"
    assert body["thread_id"]


def test_ask_keeps_thread_id(monkeypatch):
    with make_client(monkeypatch) as client:
        r = client.post("/ask", json={"question": "And the extension?", "thread_id": "abc"})
    assert r.json()["thread_id"] == "abc"


def test_ask_rejects_empty_question(monkeypatch):
    with make_client(monkeypatch) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_maps_rate_limit_to_429(monkeypatch):
    def limited(q, t):
        raise Exception("429 RESOURCE_EXHAUSTED")
    with make_client(monkeypatch, limited) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 429
