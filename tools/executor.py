"""
Tool Execution Layer — sandboxed Python runner.

Three layers of control:
  1. _patch_freq_aliases() — rewrites deprecated Pandas 3.x frequency strings
  2. exec(code, namespace) — restricts available names
  3. redirect_stdout/stderr — captures output as strings

KEY RULE for LangGraph Annotated[list, operator.add] fields:
  Return ONLY the NEW items — never the full accumulated list.
  Returning the full list causes exponential bloat each iteration.

Fix: DATASET_PATH injected into sandbox namespace so LLM code can do
     pd.read_csv(DATASET_PATH) without ever hardcoding the path.
"""

from __future__ import annotations
import io
import re
import traceback
import contextlib
from pathlib import Path
from config.settings import OUTPUTS_DIR

Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

# ── Deprecated → current Pandas 3.x frequency aliases ────────────────────────
_FREQ_ALIASES = {
    r"freq\s*=\s*['\"]M['\"]":  "freq='ME'",
    r"freq\s*=\s*['\"]Q['\"]":  "freq='QE'",
    r"freq\s*=\s*['\"]A['\"]":  "freq='YE'",
    r"freq\s*=\s*['\"]Y['\"]":  "freq='YE'",
    r"freq\s*=\s*['\"]BM['\"]": "freq='BME'",
    r"freq\s*=\s*['\"]BA['\"]": "freq='BYE'",
    r"freq\s*=\s*['\"]BY['\"]": "freq='BYE'",
    r"freq\s*=\s*['\"]AS['\"]": "freq='YS'",
    r"freq\s*=\s*['\"]H['\"]":  "freq='h'",
    r"freq\s*=\s*['\"]T['\"]":  "freq='min'",
    r"freq\s*=\s*['\"]S['\"]":  "freq='s'",
    r"freq\s*=\s*['\"]L['\"]":  "freq='ms'",
    r"freq\s*=\s*['\"]U['\"]":  "freq='us'",
    r"freq\s*=\s*['\"]N['\"]":  "freq='ns'",
    r"\.resample\(['\"]M['\"]\)": ".resample('ME')",
    r"\.resample\(['\"]Q['\"]\)": ".resample('QE')",
    r"\.resample\(['\"]A['\"]\)": ".resample('YE')",
    r"\.resample\(['\"]Y['\"]\)": ".resample('YE')",
    r"\.resample\(['\"]H['\"]\)": ".resample('h')",
    r"\.resample\(['\"]T['\"]\)": ".resample('min')",
    r"\.resample\(['\"]S['\"]\)": ".resample('s')",
}


def _patch_freq_aliases(code: str) -> str:
    """Replace deprecated Pandas frequency strings before exec()."""
    for pattern, replacement in _FREQ_ALIASES.items():
        code = re.sub(pattern, replacement, code, flags=re.IGNORECASE)
    return code


def _build_namespace(dataset_path: str = "") -> dict:
    """
    Pre-populated globals for the sandbox.
    DATASET_PATH is injected so LLM code uses the variable, not a hardcoded path.
    __saved_figures is a mutable spy list — save_fig() appends to it.
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # no GUI needed
    import matplotlib.pyplot as plt

    saved_figures: list[str] = []

    def save_fig(name: str) -> str:
        path = str(Path(OUTPUTS_DIR) / f"{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_figures.append(path)
        return path

    return {
        "pd":              pd,
        "np":              np,
        "plt":             plt,
        "Path":            Path,
        "save_fig":        save_fig,
        "OUTPUTS_DIR":     OUTPUTS_DIR,
        "DATASET_PATH":    dataset_path,   # ← LLM uses this variable to load data
        "__saved_figures": saved_figures,  # ← spy list read after exec()
    }


def run_code(code: str, dataset_path: str = "") -> dict:
    """
    Execute code in the sandboxed namespace.
    Returns: {stdout, stderr, figures, success}
    """
    if not code or not code.strip():
        return {"stdout": "", "stderr": "No code provided.", "figures": [], "success": False}

    code = _patch_freq_aliases(code)     # patch before exec

    namespace  = _build_namespace(dataset_path=dataset_path)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<analyst_sandbox>", "exec"), namespace)  # noqa: S102
        success = True
    except Exception:
        stderr_buf.write(traceback.format_exc())
        success = False

    return {
        "stdout":  stdout_buf.getvalue(),
        "stderr":  stderr_buf.getvalue(),
        "figures": namespace["__saved_figures"],
        "success": success,
    }


def tool_executor(state) -> dict:
    """
    LangGraph node — runs queued code.
    Returns ONLY NEW items for Annotated[list] fields.
    """
    code         = state.get("code_to_run", "")
    dataset_path = state.get("dataset_path", "")

    if not code:
        return {}

    print(f"\n[ToolExecutor] Running code ({len(code)} chars)...")
    result = run_code(code, dataset_path=dataset_path)

    status = "✓" if result["success"] else "✗"
    print(f"[ToolExecutor] {status}  stdout={len(result['stdout'])} chars  "
          f"figures={result['figures']}")

    if not result["success"]:
        print(f"[ToolExecutor] stderr: {result['stderr'][:400]}")
        return {
            "code_to_run": "",
            "error":       result["stderr"][:500],
            "messages":    [{"role": "tool_executor",
                             "content": f"FAILED:\n{result['stderr'][:300]}"}],
        }

    tool_entry = {
        "code":    code,
        "stdout":  result["stdout"],
        "stderr":  result["stderr"],
        "figures": result["figures"],
        "success": result["success"],
    }

    # Deduplicate: return only genuinely new figure paths
    existing_plots = set(state.get("plots", []))
    new_figs = [f for f in result["figures"] if f not in existing_plots]

    return {
        "code_to_run":  "",
        "tool_results": [tool_entry],          # Annotated → LangGraph appends this one entry
        "plots":        new_figs,              # Annotated → LangGraph appends only new paths
        "messages":     [{"role": "tool_executor",
                          "content": f"success=True stdout={result['stdout'][:200]} figures={result['figures']}"}],
    }
