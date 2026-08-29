"""Tests for API routes.

These tests require FastAPI. They are skipped if FastAPI is not installed.
"""

import uuid

import pytest

# Check if FastAPI is available
try:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI is not installed",
)


# Only import app if FastAPI is available
if FASTAPI_AVAILABLE:
    from mathmodel.main import app


@pytest.fixture
async def client():
    """Create an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data


class TestConfigEndpoint:
    """Test configuration endpoint."""

    @pytest.mark.asyncio
    async def test_get_config(self, client):
        response = await client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert data["app_name"] == "MathModel AI"
        assert "app_version" in data
        assert "default_provider" in data
        assert "model_router_enabled" in data
        assert "available_providers" in data


class TestProvidersEndpoint:
    """Test providers endpoint."""

    @pytest.mark.asyncio
    async def test_list_providers(self, client):
        response = await client.get("/api/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        provider_names = [p["name"] for p in data]
        assert "openai" in provider_names
        assert "google" in provider_names
        assert "anthropic" in provider_names
        assert "mock" in provider_names


class TestProblemCRUD:
    """Test ProblemState CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_create_problem(self, client, db_session):
        """Test creating a problem."""
        response = await client.post(
            "/api/v1/problems",
            json={
                "title": "Test Problem",
                "competition": "Test Competition",
                "raw_problem": "This is a test problem.",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Problem"
        assert data["competition"] == "Test Competition"
        assert data["current_stage"] == "ingest"
        assert data["status"] == "pending"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_problem_minimal(self, client, db_session):
        """Test creating a problem with minimal fields."""
        response = await client.post(
            "/api/v1/problems",
            json={},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["current_stage"] == "ingest"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_problems(self, client, db_session):
        """Test listing problems."""
        await client.post(
            "/api/v1/problems",
            json={"title": "Problem 1"},
        )

        response = await client.get("/api/v1/problems")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_list_problems_pagination(self, client, db_session):
        """Test pagination parameters."""
        response = await client.get("/api/v1/problems?skip=0&limit=10")
        assert response.status_code == 200

        response = await client.get("/api/v1/problems?skip=0&limit=101")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_problem(self, client, db_session):
        """Test getting a specific problem."""
        create_resp = await client.post(
            "/api/v1/problems",
            json={"title": "Get Test"},
        )
        problem_id = create_resp.json()["id"]

        response = await client.get(f"/api/v1/problems/{problem_id}")
        assert response.status_code == 200
        assert response.json()["title"] == "Get Test"

    @pytest.mark.asyncio
    async def test_get_problem_not_found(self, client, db_session):
        """Test getting a non-existent problem."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/api/v1/problems/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_problem(self, client, db_session):
        """Test updating a problem."""
        create_resp = await client.post(
            "/api/v1/problems",
            json={"title": "Original Title"},
        )
        problem_id = create_resp.json()["id"]

        response = await client.patch(
            f"/api/v1/problems/{problem_id}",
            json={"title": "Updated Title", "current_stage": "understand"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["current_stage"] == "understand"

    @pytest.mark.asyncio
    async def test_update_problem_not_found(self, client, db_session):
        """Test updating a non-existent problem."""
        fake_id = str(uuid.uuid4())
        response = await client.patch(
            f"/api/v1/problems/{fake_id}",
            json={"title": "Nope"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_problem(self, client, db_session):
        """Test deleting a problem."""
        create_resp = await client.post(
            "/api/v1/problems",
            json={"title": "Delete Me"},
        )
        problem_id = create_resp.json()["id"]

        response = await client.delete(f"/api/v1/problems/{problem_id}")
        assert response.status_code == 204

        get_resp = await client.get(f"/api/v1/problems/{problem_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_problem_not_found(self, client, db_session):
        """Test deleting a non-existent problem."""
        fake_id = str(uuid.uuid4())
        response = await client.delete(f"/api/v1/problems/{fake_id}")
        assert response.status_code == 404