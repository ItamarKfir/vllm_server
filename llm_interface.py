import os
import uuid
from typing import AsyncGenerator

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from config_manager import setting
from logger import log_with_timestamp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONHASHSEED"] = "0"


class LLMInterface:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.init_engine()
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance to allow re-initialization with new config."""
        if cls._instance is not None:
            cls._instance._initialized = False
            cls._instance = None
    
    def init_engine(self):
        log_with_timestamp("🚀 Initializing Async vLLM engine...")

        engine_args = AsyncEngineArgs(
            model=setting.llm.llm_models[0],
            max_num_seqs=setting.llm.max_num_seqs,
            max_num_batched_tokens=setting.llm.max_num_batched_tokens,
            gpu_memory_utilization=setting.llm.gpu_util,
            kv_cache_memory_bytes=int(
                setting.llm.gpu_kv_cache_gb * (1024 ** 3)
            ),
            max_model_len=setting.llm.max_model_len,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.sampling_params = SamplingParams(
            temperature=setting.llm.llm_temperature,
            max_tokens=setting.llm.max_output_tokens,
        )

        log_with_timestamp("✅ Async vLLM engine ready!")
        self._initialized = True


    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())
        previous_text = ""

        async for output in self.engine.generate(
            prompt,
            self.sampling_params,
            request_id,
        ):
            text = output.outputs[0].text
            delta = text[len(previous_text):]
            previous_text = text

            if delta:
                yield delta

    async def generate(self, prompt: str) -> str:
        """Run full generation without streaming; returns complete text."""
        request_id = str(uuid.uuid4())
        full_text = ""
        async for output in self.engine.generate(
            prompt,
            self.sampling_params,
            request_id,
        ):
            full_text = output.outputs[0].text
        return full_text
