"""
Tools for the LlamaIndex agent
"""
from typing import Optional, List
import json
import os
import glob
import sys
from datetime import datetime
from llama_index.core.tools import FunctionTool



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


class FileSearchTool:
    """Tool for searching and reading files in the current folder"""
    
    def __init__(self, base_path: str = "."):
        self.name = "file_search"
        # Normalize and get absolute path
        self.base_path = os.path.abspath(os.path.expanduser(base_path))
        print(f"[FileSearchTool] Initialized with base_path: {self.base_path}", flush=True)  # Debug
        sys.stdout.flush()
        self.supported_extensions = ['.txt', '.py', '.json', '.ini', '.cfg', '.conf']
        self.description = """
        Searches for and reads files in the project directory (/home/itamar/Desktop/vllm_server).
        Supports file types: TXT, PY, JSON, INI, CFG, CONF
        
        IMPORTANT: Input must be a JSON string with 'action' and parameters.
        
        To search for a file by name, use: {"action": "search", "filename": "filename.ext"}
        To read a file, use: {"action": "read", "filepath": "filename.ext"}
        To list all files, use: {"action": "list"}
        
        Examples:
        - To find routes.py: {"action": "search", "filename": "routes.py"}
        - To read routes.py: {"action": "read", "filepath": "routes.py"}
        - To find all Python files: {"action": "search", "pattern": "*.py"}
        """
    
    def _is_supported_file(self, filepath: str) -> bool:
        """Check if file has a supported extension"""
        _, ext = os.path.splitext(filepath.lower())
        return ext in self.supported_extensions
    
    def _get_file_size(self, filepath: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except:
            return 0
    
    def _read_file_content(self, filepath: str, max_size: int = 100000) -> str:
        """Read file content with size limit"""
        try:
            file_size = self._get_file_size(filepath)
            if file_size > max_size:
                return f"[File too large: {file_size} bytes. Showing first {max_size} bytes...]\n\n"
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"[Error reading file: {str(e)}]"
    
    def execute(self, command: str) -> str:
        """Execute a file search or read operation"""
        try:
            # Print to server console for debugging (with flush)
            print(f"\n[FileSearchTool] Execute called with command: {command}", flush=True)
            print(f"[FileSearchTool] Base search path: {self.base_path}", flush=True)
            print(f"[FileSearchTool] Base path exists: {os.path.exists(self.base_path)}", flush=True)
            sys.stdout.flush()
            
            # Debug: Print base path
            debug_info = f"[DEBUG] Base search path: {self.base_path}\n"
            debug_info += f"[DEBUG] Base path exists: {os.path.exists(self.base_path)}\n"
            
            # Parse JSON command - try to handle both JSON and plain filenames
            if isinstance(command, str):
                # First, try to parse as JSON
                try:
                    cmd = json.loads(command)
                    print(f"[FileSearchTool] Parsed JSON command: {cmd}", flush=True)
                    sys.stdout.flush()
                except json.JSONDecodeError:
                    # If not JSON, treat as a filename and create a search command
                    print(f"[FileSearchTool] Input is not JSON, treating as filename: {command}", flush=True)
                    sys.stdout.flush()
                    # Remove quotes if present
                    filename = command.strip().strip('"').strip("'")
                    # Auto-detect: if it looks like a file path/name, search for it
                    if filename:
                        cmd = {"action": "search", "filename": filename}
                        print(f"[FileSearchTool] Auto-created search command: {cmd}", flush=True)
                        sys.stdout.flush()
                    else:
                        return f"Error: Invalid input format. Expected JSON or filename. Received: {command}\n{debug_info}"
            else:
                cmd = command
            
            action = cmd.get("action", "").lower()
            print(f"[FileSearchTool] Action: {action}", flush=True)
            sys.stdout.flush()
            
            if action == "search":
                # Search for files by pattern or filename
                pattern = cmd.get("pattern", "")
                filename = cmd.get("filename", "")
                
                print(f"[FileSearchTool] Search - pattern: '{pattern}', filename: '{filename}'", flush=True)
                sys.stdout.flush()
                debug_info += f"[DEBUG] Search action - pattern: '{pattern}', filename: '{filename}'\n"
                
                if pattern:
                    # Use glob pattern
                    search_path = os.path.join(self.base_path, "**", pattern)
                    print(f"[FileSearchTool] Searching with glob: {search_path}", flush=True)
                    sys.stdout.flush()
                    debug_info += f"[DEBUG] Searching with glob pattern: {search_path}\n"
                    matches = glob.glob(search_path, recursive=True)
                    print(f"[FileSearchTool] Found {len(matches)} matches with glob", flush=True)
                    sys.stdout.flush()
                    debug_info += f"[DEBUG] Found {len(matches)} matches with glob\n"
                elif filename:
                    # Search by filename (partial match)
                    matches = []
                    # Also check if it's an exact filename match (with or without extension)
                    filename_lower = filename.lower()
                    print(f"[FileSearchTool] Searching for filename: '{filename}' (case-insensitive)", flush=True)
                    sys.stdout.flush()
                    debug_info += f"[DEBUG] Searching for filename: '{filename}' (case-insensitive)\n"
                    debug_info += f"[DEBUG] Walking directory: {self.base_path}\n"
                    
                    if not os.path.exists(self.base_path):
                        error_msg = f"Error: Base search path does not exist: {self.base_path}\n{debug_info}"
                        print(f"[FileSearchTool] ERROR: {error_msg}", flush=True)
                        sys.stdout.flush()
                        return error_msg
                    
                    walk_count = 0
                    for root, dirs, files in os.walk(self.base_path):
                        walk_count += 1
                        # Skip hidden directories and common ignore patterns
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.git']]
                        for file in files:
                            file_lower = file.lower()
                            # Check if filename matches (exact or partial) and is a supported file type
                            if (filename_lower in file_lower or file_lower.startswith(filename_lower)) and self._is_supported_file(file):
                                full_match_path = os.path.join(root, file)
                                matches.append(full_match_path)
                                print(f"[FileSearchTool] Found match: {full_match_path}", flush=True)
                                sys.stdout.flush()
                                debug_info += f"[DEBUG] Found match: {full_match_path}\n"
                    
                    print(f"[FileSearchTool] Walked {walk_count} directories, found {len(matches)} total matches", flush=True)
                    sys.stdout.flush()
                else:
                    error_msg = f"Error: Provide either 'pattern' or 'filename' for search\n{debug_info}"
                    print(f"[FileSearchTool] ERROR: {error_msg}")
                    return error_msg
                
                # Filter to supported file types and get relative paths
                supported_matches = []
                for match in matches:
                    if os.path.isfile(match) and self._is_supported_file(match):
                        rel_path = os.path.relpath(match, self.base_path)
                        file_size = self._get_file_size(match)
                        supported_matches.append({
                            'path': rel_path,
                            'size': file_size,
                            'full_path': match
                        })
                
                if not supported_matches:
                    # List some files that were checked
                    checked_files = []
                    try:
                        for root, dirs, files in os.walk(self.base_path):
                            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.git']]
                            for file in files[:10]:  # Limit to 10 examples
                                if self._is_supported_file(file):
                                    checked_files.append(os.path.relpath(os.path.join(root, file), self.base_path))
                    except:
                        pass
                    
                    error_msg = f"No matching files found for pattern '{pattern or filename}'\n"
                    error_msg += f"Searched in base path: {self.base_path}\n"
                    error_msg += f"Base path exists: {os.path.exists(self.base_path)}\n"
                    if checked_files:
                        error_msg += f"Example files found in directory (first 10): {', '.join(checked_files)}\n"
                    error_msg += debug_info
                    print(f"[FileSearchTool] {error_msg}", flush=True)
                    sys.stdout.flush()
                    return error_msg
                
                # If exactly one file found, automatically read and return its content
                if len(supported_matches) == 1:
                    match = supported_matches[0]
                    filepath = match['path']
                    full_path = match['full_path']
                    
                    print(f"[FileSearchTool] Single file found, reading content: {filepath}", flush=True)
                    sys.stdout.flush()
                    
                    # Read the file content
                    content = self._read_file_content(full_path)
                    file_size = self._get_file_size(full_path)
                    
                    result = f"Found and read file: {filepath}\n"
                    result += f"Full path: {full_path}\n"
                    result += f"Size: {file_size} bytes\n"
                    result += f"Type: {os.path.splitext(full_path)[1]}\n"
                    result += f"\n--- File Content ---\n{content}\n--- End of file ---"
                    
                    print(f"[FileSearchTool] Returning file content ({file_size} bytes)", flush=True)
                    sys.stdout.flush()
                    return result
                
                # Multiple files found - return list
                result = f"Found {len(supported_matches)} file(s):\n"
                result += debug_info + "\n"
                for match in supported_matches[:20]:  # Limit to 20 results
                    result += f"  - {match['path']} (Full path: {match['full_path']}, {match['size']} bytes)\n"
                
                if len(supported_matches) > 20:
                    result += f"\n... and {len(supported_matches) - 20} more files"
                
                result += f"\n\nTo read a specific file, use: {{\"action\": \"read\", \"filepath\": \"filename.ext\"}}"
                
                print(f"[FileSearchTool] Returning {len(supported_matches)} matches", flush=True)
                sys.stdout.flush()
                return result
            
            elif action == "read":
                # Read a specific file
                filepath = cmd.get("filepath", "")
                if not filepath:
                    error_msg = "Error: 'filepath' is required for read action"
                    print(f"[FileSearchTool] ERROR: {error_msg}", flush=True)
                    sys.stdout.flush()
                    return error_msg
                
                print(f"[FileSearchTool] Read action - filepath: '{filepath}'", flush=True)
                sys.stdout.flush()
                debug_info += f"[DEBUG] Read action - filepath: '{filepath}'\n"
                
                # Construct full path
                full_path = os.path.join(self.base_path, filepath)
                full_path = os.path.normpath(full_path)
                abs_full_path = os.path.abspath(full_path)
                
                print(f"[FileSearchTool] Base path: {self.base_path}", flush=True)
                print(f"[FileSearchTool] Requested filepath: {filepath}", flush=True)
                print(f"[FileSearchTool] Constructed full path: {full_path}", flush=True)
                print(f"[FileSearchTool] Absolute full path: {abs_full_path}", flush=True)
                print(f"[FileSearchTool] Path exists: {os.path.exists(abs_full_path)}", flush=True)
                print(f"[FileSearchTool] Is file: {os.path.isfile(abs_full_path) if os.path.exists(abs_full_path) else 'N/A'}", flush=True)
                sys.stdout.flush()
                
                debug_info += f"[DEBUG] Base path: {self.base_path}\n"
                debug_info += f"[DEBUG] Requested filepath: {filepath}\n"
                debug_info += f"[DEBUG] Constructed full path: {full_path}\n"
                debug_info += f"[DEBUG] Absolute full path: {abs_full_path}\n"
                debug_info += f"[DEBUG] Path exists: {os.path.exists(abs_full_path)}\n"
                debug_info += f"[DEBUG] Is file: {os.path.isfile(abs_full_path) if os.path.exists(abs_full_path) else 'N/A'}\n"
                
                # Security check - ensure path is within base_path
                abs_base_path = os.path.abspath(self.base_path)
                if not abs_full_path.startswith(abs_base_path):
                    return f"Error: Path outside allowed directory\n{debug_info}\nTried to access: {abs_full_path}\nAllowed base: {abs_base_path}"
                
                if not os.path.exists(abs_full_path):
                    return f"Error: File '{filepath}' does not exist\n{debug_info}\nTried to open: {abs_full_path}\nBase search path: {abs_base_path}"
                
                if not os.path.isfile(abs_full_path):
                    return f"Error: '{filepath}' is not a file\n{debug_info}\nPath: {abs_full_path}"
                
                if not self._is_supported_file(abs_full_path):
                    ext = os.path.splitext(abs_full_path)[1]
                    return f"Error: File type '{ext}' not supported. Supported: {', '.join(self.supported_extensions)}\n{debug_info}\nTried to open: {abs_full_path}"
                
                # Read file content
                content = self._read_file_content(abs_full_path)
                file_size = self._get_file_size(abs_full_path)
                
                result = f"File: {filepath}\n"
                result += f"Full path: {abs_full_path}\n"
                result += f"Size: {file_size} bytes\n"
                result += f"Type: {os.path.splitext(abs_full_path)[1]}\n"
                result += f"\n--- Content ---\n{content}\n--- End of file ---"
                
                return result
            
            elif action == "list":
                # List all supported files in current directory
                matches = []
                for root, dirs, files in os.walk(self.base_path):
                    # Skip hidden directories and common ignore patterns
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.git']]
                    
                    for file in files:
                        if self._is_supported_file(file):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, self.base_path)
                            file_size = self._get_file_size(full_path)
                            matches.append({
                                'path': rel_path,
                                'size': file_size
                            })
                
                if not matches:
                    return "No supported files found in current directory"
                
                result = f"Found {len(matches)} supported file(s):\n"
                for match in sorted(matches, key=lambda x: x['path']):
                    result += f"  - {match['path']} ({match['size']} bytes)\n"
                
                return result
            
            else:
                return f"Unknown action: {action}. Available actions: search, read, list"
                
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing command JSON: {str(e)}. Expected JSON format.\nReceived command: {command}\n{debug_info}"
            print(f"[FileSearchTool] JSON ERROR: {error_msg}", flush=True)
            sys.stdout.flush()
            return error_msg
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}\n{debug_info}"
            print(f"[FileSearchTool] EXCEPTION: {error_msg}", flush=True)
            sys.stdout.flush()
            return error_msg


def get_function_tools(tool_usage_tracker):
    import json
    import sys
    
    tool_instances = [
        CalculatorTool(),
        WebSearchTool(),
        FileOperationsTool(base_path="/tmp"),
        FileSearchTool(base_path="/home/itamar/Desktop/vllm_server"),
    ]
    
    function_tools = []
    for tool in tool_instances:
        def create_tool_function(tool_instance):
            def tool_function(*args, **kwargs) -> str:
                tool_name = tool_instance.name
                if tool_name not in tool_usage_tracker["tools_used"]:
                    tool_usage_tracker["tools_used"].append(tool_name)
                
                input_data = None
                if kwargs:
                    if 'args' in kwargs:
                        args_value = kwargs['args']
                        if hasattr(args_value, 'keys'):
                            cmd_dict = dict(args_value)
                        else:
                            cmd_dict = dict(args_value) if isinstance(args_value, dict) else vars(args_value)
                        input_data = json.dumps(cmd_dict)
                    else:
                        input_data = json.dumps(kwargs)
                elif args:
                    input_data = args[0] if isinstance(args[0], str) else str(args[0])
                else:
                    input_data = ""
                
                try:
                    result = tool_instance.execute(input_data)
                    return result
                except Exception as e:
                    import traceback
                    return f"Error in tool '{tool_name}': {str(e)}\n{traceback.format_exc()}"
            return tool_function
        
        tool_func = create_tool_function(tool)
        tool_func.__name__ = tool.name
        function_tool = FunctionTool.from_defaults(fn=tool_func, name=tool.name, description=tool.description)
        function_tools.append(function_tool)
    
    return function_tools