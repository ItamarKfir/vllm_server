"""
Tools for the LlamaIndex agent
"""
from typing import Optional
import json
import os
from datetime import datetime


class CalculatorTool:
    """Tool for performing mathematical calculations"""
    
    def __init__(self):
        self.name = "calculator"
        self.description = """
        Performs mathematical calculations. 
        Input should be a valid Python mathematical expression.
        Examples: "2 + 2", "10 * 5", "sqrt(16)", "2 ** 3"
        """
    
    def execute(self, expression: str) -> str:
        """Execute a mathematical calculation"""
        try:
            # Safe evaluation - only allow basic math operations
            allowed_names = {
                k: v for k, v in __builtins__.items() if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
            }
            allowed_names.update({
                'sqrt': lambda x: x ** 0.5,
                'sin': lambda x: __import__('math').sin(x),
                'cos': lambda x: __import__('math').cos(x),
                'tan': lambda x: __import__('math').tan(x),
                'pi': __import__('math').pi,
                'e': __import__('math').e,
            })
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"


class WebSearchTool:
    """Tool for searching the web (simulated)"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = """
        Searches the web for information.
        Input should be a search query string.
        Note: This is a simulated search tool for demonstration.
        """
        # Simulated search results database
        self.search_db = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "llama index": "LlamaIndex is a framework for building LLM applications with data integration.",
            "fastapi": "FastAPI is a modern web framework for building APIs with Python.",
            "vllm": "vLLM is a high-throughput LLM inference and serving engine.",
        }
    
    def execute(self, query: str) -> str:
        """Execute a web search (simulated)"""
        query_lower = query.lower()
        
        # Try to find matching results
        results = []
        for key, value in self.search_db.items():
            if key in query_lower:
                results.append(f"Found: {key} - {value}")
        
        if results:
            return "\n".join(results)
        else:
            return f"Search results for '{query}': No specific results found. This is a simulated search tool."


class FileOperationsTool:
    """Tool for file operations"""
    
    def __init__(self, base_path: str = "."):
        self.name = "file_operations"
        self.base_path = base_path
        self.description = """
        Performs file operations like reading, writing, and listing files.
        Input should be a JSON string with 'operation' and 'path' fields.
        Operations: 'read', 'write', 'list', 'exists'
        Example: {"operation": "read", "path": "test.txt"}
        """
    
    def execute(self, command: str) -> str:
        """Execute a file operation"""
        try:
            # Parse JSON command
            if isinstance(command, str):
                cmd = json.loads(command)
            else:
                cmd = command
            
            operation = cmd.get("operation")
            path = cmd.get("path", "")
            
            # Ensure path is within base_path for security
            full_path = os.path.join(self.base_path, path)
            full_path = os.path.normpath(full_path)
            
            if not full_path.startswith(os.path.abspath(self.base_path)):
                return "Error: Path outside allowed directory"
            
            if operation == "read":
                if not os.path.exists(full_path):
                    return f"Error: File '{path}' does not exist"
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File content of '{path}':\n{content}"
            
            elif operation == "write":
                content = cmd.get("content", "")
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Successfully wrote to '{path}'"
            
            elif operation == "list":
                if not os.path.exists(full_path):
                    return f"Error: Path '{path}' does not exist"
                if os.path.isdir(full_path):
                    items = os.listdir(full_path)
                    return f"Directory contents of '{path}':\n" + "\n".join(items)
                else:
                    return f"'{path}' is not a directory"
            
            elif operation == "exists":
                exists = os.path.exists(full_path)
                return f"File '{path}' exists: {exists}"
            
            else:
                return f"Unknown operation: {operation}. Available: read, write, list, exists"
                
        except json.JSONDecodeError as e:
            return f"Error parsing command JSON: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


def get_all_tools():
    """Get all available tools"""
    return [
        CalculatorTool(),
        WebSearchTool(),
        FileOperationsTool(base_path="/tmp"),  # Safe base path
    ]
