
from __future__ import annotations

from typing import Dict, List, Optional

from gsm8k_multiagent.config import (
    APIConfig,
    LOCAL_MODEL_NAME,
    STRONG_AGENT_CONFIG,
    WEAK_AGENT_CONFIG,
)


# ---------------------------------------------------------------------------
# Remote client
# ---------------------------------------------------------------------------

class RemoteAPIClient:

    def __init__(self, config: APIConfig):
        self._config = config
        try:
            from openai import OpenAI
            self._client   = OpenAI(base_url=config.api_url, api_key=config.api_key)
            self._available = True
        except ImportError:
            self._available = False
            print("[RemoteAPIClient] 'openai' package not found; stub responses will be returned.")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if not self._available:
            return "[STUB] openai package not installed."
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Local HuggingFace client (offline weak agent)
# ---------------------------------------------------------------------------

class LocalModelClient:

    def __init__(self, model_name: str = LOCAL_MODEL_NAME):
        self._model_name = model_name
        self._pipe       = None

    def load(self) -> bool:
    
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            dtype  = torch.float16 if device == 0 else torch.float32
            self._pipe = pipeline(
                "text-generation",
                model=self._model_name,
                device=device,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if self._pipe.tokenizer.pad_token is None:
                self._pipe.tokenizer.pad_token = self._pipe.tokenizer.eos_token
            print(f"[LocalModelClient] Loaded '{self._model_name}' on device={device}.")
            return True
        except Exception as exc:
            print(f"[LocalModelClient] Load failed: {exc}")
            return False

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if self._pipe is None:
            return "[LocalModel] Model not loaded."
        prompt = self._to_llama_prompt(messages)
        out = self._pipe(
            prompt,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        return out[0]['generated_text'].strip() if out else ""

    @staticmethod
    def _to_llama_prompt(messages: List[Dict[str, str]]) -> str:
        """Serialise a chat history to the Llama-3.2 ``<|…|>`` prompt format."""
        parts = []
        for msg in messages:
            role, content = msg['role'], msg['content']
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Dual-agent facade
# ---------------------------------------------------------------------------

class DualAgentClient:

    def __init__(
        self,
        strong_config: APIConfig = STRONG_AGENT_CONFIG,
        weak_config:   APIConfig = WEAK_AGENT_CONFIG,
        use_local_weak: bool = False,
        local_model_name: Optional[str] = None,
    ):
        self._strong = RemoteAPIClient(strong_config)

        if use_local_weak:
            local = LocalModelClient(local_model_name or LOCAL_MODEL_NAME)
            local.load()
            self._weak: RemoteAPIClient | LocalModelClient = local
        else:
            self._weak = RemoteAPIClient(weak_config)

    def call(self, messages: List[Dict[str, str]], agent: str = "weak") -> str:
        client = self._strong if agent == "strong" else self._weak
        return client.generate(messages)
