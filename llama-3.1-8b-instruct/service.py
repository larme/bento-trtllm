import json
import os
import random
import subprocess
from typing import AsyncGenerator, Optional

import bentoml
import fastapi
import numpy as np
from annotated_types import Ge, Le
from typing_extensions import Annotated


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MAX_TOKENS = 2048
MAX_NEW_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

runtime_image = bentoml.images.PythonImage(base_image="docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", lock_python_packages=False)\
                              .run("apt-get -y update && apt-get -y install libopenmpi-dev git python3-pip")\
                              .requirements_file("requirements.txt")

openai_api_app = fastapi.FastAPI()

@bentoml.asgi_app(openai_api_app, path="/")
@bentoml.service(
    name="bentotrtllm-llama3.1-8b-insruct-service",
    image=runtime_image,
    envs=[{'name': 'HF_TOKEN'}, {"name": "UV_INDEX_STRATEGY", "value": "unsafe-best-match"}],
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class TRTLLM:

    hf_model = bentoml.models.HuggingFaceModel(MODEL_ID, exclude=['*.pth', '*.pt'])

    @bentoml.on_startup
    async def init(self) -> None:
        from transformers import AutoTokenizer
        from tensorrt_llm.llmapi import LLM, BuildConfig, KvCacheConfig, SamplingParams
        from openai_server import OpenAIServer

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.85)
        self.build_config = BuildConfig(max_batch_size=256, max_seq_len=MAX_TOKENS)

        self.llm = LLM(
            self.hf_model,
            tokenizer=self.hf_model,
            build_config=self.build_config,
            kv_cache_config=self.kv_cache_config,
        )
        self.server = OpenAIServer(
            self.llm,
            MODEL_ID,
            self.tokenizer,
            openai_api_app
        )

    @bentoml.on_shutdown
    async def quit_llm(self):
        self.llm.shutdown()

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_NEW_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from tensorrt_llm import SamplingParams

        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)
        sampling_params = SamplingParams(max_new_tokens=max_tokens)

        promise = self.llm.generate_async(
            prompt,
            streaming=True,
            sampling_params=sampling_params
        )

        async for output in promise:
            yield output.outputs[0].text_diff
