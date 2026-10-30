"""
Central configuration for the agentic pipeline.
Set your GROQ_API_KEY in a .env file or export as environment variable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "openai/gpt-oss-120b"       # current recommended Groq model

# ── Pipeline limits ────────────────────────────────────────────────────────────
MAX_ITERATIONS = 8                               # hard ceiling on agent loop iterations
VALID_NODES    = ["analyst", "researcher", "writer"]

# ── Tool execution ─────────────────────────────────────────────────────────────
PYTHON_TIMEOUT_SECONDS = 30

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR = "outputs"
