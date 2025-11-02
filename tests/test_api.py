"""
API endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


# Note: The following tests require properly configured API keys and indexed documents
# Uncomment and modify as needed for your environment

# def test_query_endpoint():
#     """Test query endpoint"""
#     response = client.post(
#         "/api/v1/query",
#         json={
#             "question": "What was the revenue in Q1 2024?",
#             "top_k": 5
#         }
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert "answer" in data
#     assert "sources" in data
#     assert "retrieved_docs" in data


# def test_chat_endpoint():
#     """Test chat endpoint with history"""
#     response = client.post(
#         "/api/v1/chat",
#         json={
#             "question": "What about the operating margin?",
#             "chat_history": [
#                 {
#                     "question": "What was the revenue?",
#                     "answer": "The revenue was $450 million."
#                 }
#             ],
#             "top_k": 5
#         }
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert "answer" in data


# def test_stats_endpoint():
#     """Test stats endpoint"""
#     response = client.get("/api/v1/stats")
#     assert response.status_code == 200
#     data = response.json()
#     assert "total_vector_count" in data
#     assert "dimension" in data
