"""
Unit tests for agent functionality and requests
"""
import pytest
import requests
import os
import time
import json

# Configuration
AGENT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://localhost:8001")


class TestAgentRequests:
    """Test agent chat requests"""
    
    @pytest.fixture(scope="class")
    def wait_for_server(self):
        """Wait for agent server to be ready"""
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f"{AGENT_SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        return False
    
    def test_simple_chat(self, wait_for_server):
        """Test simple chat request"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": "Hello, what is 2 + 2?"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        print(f"\nAgent response: {data['response']}")
    
    def test_calculator_tool(self, wait_for_server):
        """Test calculator tool usage"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": "Calculate 15 * 23 using the calculator tool"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Should contain the result or mention calculation
        assert len(data["response"]) > 0
        print(f"\nCalculator response: {data['response']}")
    
    def test_web_search_tool(self, wait_for_server):
        """Test web search tool usage"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": "Search for information about Python programming"},
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        print(f"\nWeb search response: {data['response']}")
    
    def test_file_operations_tool(self, wait_for_server):
        """Test file operations tool"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        # First, ask to write a file
        write_response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={
                "message": "Write a test file with content 'Hello from agent' to /tmp/test_agent.txt using file operations"
            },
            timeout=30
        )
        
        assert write_response.status_code == 200
        write_data = write_response.json()
        assert "response" in write_data
        print(f"\nFile write response: {write_data['response']}")
        
        # Then, ask to read it back
        read_response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={
                "message": "Read the file /tmp/test_agent.txt using file operations"
            },
            timeout=30
        )
        
        assert read_response.status_code == 200
        read_data = read_response.json()
        assert "response" in read_data
        print(f"\nFile read response: {read_data['response']}")
    
    def test_complex_query(self, wait_for_server):
        """Test complex query using multiple tools"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={
                "message": "Calculate 10 * 5, then search for information about FastAPI, and tell me both results"
            },
            timeout=45
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        print(f"\nComplex query response: {data['response']}")
    
    def test_error_handling(self, wait_for_server):
        """Test error handling"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        # Test with empty message
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={"message": ""},
            timeout=30
        )
        
        # Should either return a response or an error, but not crash
        assert response.status_code in [200, 400, 422]
    
    def test_conversation_id(self, wait_for_server):
        """Test conversation ID handling"""
        if not wait_for_server:
            pytest.skip("Agent server is not running or not healthy")
        
        conversation_id = "test-conv-123"
        response = requests.post(
            f"{AGENT_SERVER_URL}/chat",
            json={
                "message": "Hello",
                "conversation_id": conversation_id
            },
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        # Note: Current implementation may not use conversation_id, but should accept it


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
