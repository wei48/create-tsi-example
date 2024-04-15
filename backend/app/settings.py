# SPDX-FileCopyrightText: 2024 Deutsche Telekom AG, LlamaIndex, Vercel, Inc.
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



def llm_config_from_env() -> Dict:
    from llama_index.core.constants import DEFAULT_TEMPERATURE

    model = os.getenv("MODEL")
    temperature = os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens = os.getenv("LLM_MAX_TOKENS")
    api_key = os.getenv("TSI_API_KEY")

    config = {
        "model": model,
        "api_key": api_key,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
    }
    return config


def embedding_config_from_env() -> Dict:
    model = os.getenv("EMBEDDING_MODEL")
    dimension = os.getenv("EMBEDDING_DIM")
    api_key = os.getenv("TSI_API_KEY")

    config = {
        "model": model,
        "dimensions": int(dimension) if dimension is not None else 768,
        "api_key": api_key,
    }
    return config


def init_settings():
    llm_configs = llm_config_from_env()
    embedding_configs = embedding_config_from_env()

    Settings.embed_model = OpenAIEmbedding(**embedding_configs)
    Settings.llm = OpenAI(
        **llm_configs
    )
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
