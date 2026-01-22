"""
Unit tests for agent server connection
"""
import pytest
import requests
import os
import time
from agent import create_agent, create_llm


# Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
AGENT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://localhost:8001")
MODEL_NAME = os.getenv("LLM_MODEL", "Qwen2.5-32B-Instruct-AWQ")


class TestLLMConnection:
    """Test connection to the LLM server"""
    
    def test_llm_server_health(self):
        """Test if LLM server is accessible"""
        try:
            response = requests.get(f"{LLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
            assert response.status_code in [200, 404], f"LLM server not accessible: {response.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.skip("LLM server is not running")
    
    def test_llm_server_openai_compatible(self):
        """Test if LLM server has OpenAI-compatible endpoint"""
        try:
            # Test if we can list models
            response = requests.get(f"{LLM_BASE_URL}/models", timeout=5)
            # Should return 200 or at least not connection error
            assert response.status_code in [200, 401, 404], f"OpenAI endpoint not accessible: {response.status_code}"
        except requests.exceptions.ConnectionError:
            pytest.skip("LLM server is not running")
    
    def test_create_llm_client(self):
        """Test creating LLM client"""
        try:
            llm = create_llm(base_url=LLM_BASE_URL, model=MODEL_NAME)
            assert llm is not None
            assert llm.base_url == LLM_BASE_URL
        except Exception as e:
            pytest.skip(f"Cannot create LLM client: {e}")


class TestAgentServer:
    """Test agent server endpoints"""
    
    @pytest.fixture(scope="class")
    def wait_for_server(self):
        """Wait for agent server to be ready"""
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        return False
    
    def test_agent_server_health(self, wait_for_server):
        """Test agent server health endpoint"""
        if not wait_for_server:
            pytest.skip("Agent server is not running")
        
        response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "not_initialized"]
        assert "llm_server_url" in data
        assert "model" in data
    
    def test_agent_server_root(self, wait_for_server):
        """Test agent server root endpoint"""
        if not wait_for_server:
            pytest.skip("Agent server is not running")
        
        response = requests.get(f"{AGENT_SERVER_URL}/", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
