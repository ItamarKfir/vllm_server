from pydantic import BaseModel, Field
from pathlib import Path
import json

class LLMConfig(BaseModel):
    gpu_util: float = Field(gt=0, le=1)
    gpu_kv_cache_gb: float = Field(gt=0)
    max_model_len: int = Field(gt=0)
    max_num_seqs: int = Field(gt=0)
    max_num_batched_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    llm_temperature: float = Field(ge=0, le=2)
    llm_models: list[str]

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(gt=0, le=65535, default=8000)
    reload: bool = False
    log_level: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")

class Setting:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        config_data = self._load()
        
        # Handle both old format (flat) and new format (nested)
        if "llm" in config_data:
            # New format: nested structure
            self.llm: LLMConfig = LLMConfig(**config_data.get("llm", {}))
            self.server: ServerConfig = ServerConfig(**config_data.get("server", {}))
        else:
            # Old format: flat structure (backward compatibility)
            self.llm: LLMConfig = LLMConfig(**config_data)
            self.server: ServerConfig = ServerConfig()  # Use defaults
    
    def _load(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data

    def _save_llm(self, llm_config: LLMConfig):
        """Save LLM configuration"""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        config_data["llm"] = llm_config.dict()
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        self.llm = llm_config
    
    def _save_server(self, server_config: ServerConfig):
        """Save server configuration"""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        config_data["server"] = server_config.dict()
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        self.server = server_config

setting = Setting(config_path="llm_config.json")
    