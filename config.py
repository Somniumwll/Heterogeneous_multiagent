"""
config.py – API endpoints, model identifiers, and experiment defaults.

Replace the placeholder API key strings with real values before running.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """Holds all parameters needed to call one language model endpoint."""
    api_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 1500


#: Strong agent – GPT-4o via a compatible proxy endpoint
STRONG_AGENT_CONFIG = APIConfig(
    api_url="",
    api_key="YOUR_STRONG_API_KEY",
    model="gpt-4o",
    temperature=0.1,
    max_tokens=1500,
)

#: Weak agent – Llama-3.2-1B via OpenRouter
WEAK_AGENT_CONFIG = APIConfig(
    api_url="",
    api_key="YOUR_WEAK_API_KEY",
    model="llama-3.2-1b",
    temperature=0.3,
    max_tokens=800,
)

#: Local weak agent – HuggingFace Llama-3.2-3B (offline fallback)
LOCAL_MODEL_NAME = "Llama-3.2-3B"


# ---------------------------------------------------------------------------
# Entropy thresholds for three-tier guidance  (τ₁ < τ₂)
# ---------------------------------------------------------------------------

#: H(P_w) ≤ τ₁  →  light_guidance
ENTROPY_TAU_1: float = 2.0

#: τ₁ < H(P_w) ≤ τ₂  →  moderate_guidance
ENTROPY_TAU_2: float = 3.5

#: H(P_w) > τ₂  →  intensive_guidance  (implicit)


# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE: int = 50
DEFAULT_MAX_ROUNDS: int = 5
DEFAULT_RANDOM_SEED: int = 42
