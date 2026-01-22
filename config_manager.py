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

class Setting:
    def __init__(self,config_path: str):
        self.config_path = Path(config_path)
        self.llm: LLMConfig = self._load()
    
    def _load(self) -> LLMConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        return LLMConfig(**config_data)

    def _save(self, llm_config: LLMConfig):
        with open(self.config_path, 'w') as f:
            json.dump(llm_config.dict(), f, indent=4)
        self._load()

setting = Setting(config_path="llm_config.json")
    