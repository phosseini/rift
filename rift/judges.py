"""
Judge model registry and credential resolution shared by every experiment script.

Three ways to name a judge on the command line:

1. A registered model id (config/rift.yaml ``judges.registry``), e.g. ``gpt-5.4-2026-03-05``.
   Uses the provider's direct API key from .env (OPENAI_API_KEY / GEMINI_API_KEY).
   If that key is missing but PORTKEY_API_KEY is set, the call is routed through
   Portkey instead, addressed as ``<slug>/<model>`` where the slug is the
   provider entry in your Portkey model catalog (defaults: ``@openai``,
   ``@google``; override with PORTKEY_OPENAI_PROVIDER / PORTKEY_GOOGLE_PROVIDER).

2. A Portkey model-catalog address, e.g. ``@anthropic/claude-opus-5`` or
   ``@openrouter/openai/gpt-5.6-sol``. Anything starting with ``@`` goes to
   Portkey verbatim, so any model your workspace exposes can be a judge without
   editing this file.

3. ``portkey:<model>`` for workspaces that select the upstream provider with the
   ``x-portkey-provider`` header (set PORTKEY_PROVIDER in .env) rather than the
   ``@slug/`` prefix.

Portkey settings in .env:
    PORTKEY_API_KEY   required for any Portkey route
    PORTKEY_BASE_URL  optional, default https://api.portkey.ai/v1
"""

from __future__ import annotations

import os
import sys

from . import config as rift_config
from .schema import ModelConfig

# Registry, default judge and Portkey settings come from config/rift.yaml (`judges` section).
_SLUG_ENV = {"openai": "PORTKEY_OPENAI_PROVIDER", "google": "PORTKEY_GOOGLE_PROVIDER"}


def registry() -> dict[str, dict]:
    """model id -> {provider, key_env} from config."""
    return rift_config.section("judges").get("registry") or {}


def default_judge() -> str:
    return rift_config.section("judges").get("default") or next(iter(registry()), "")


def _portkey_settings() -> dict:
    return rift_config.section("judges").get("portkey") or {}


def _portkey_available() -> bool:
    return bool(os.getenv("PORTKEY_API_KEY"))


def _portkey_config(display: str, wire_model: str, extra_headers: dict[str, str] | None = None) -> ModelConfig:
    key = os.getenv("PORTKEY_API_KEY")
    if not key:
        sys.exit(
            f"Judge '{display}' requires Portkey but PORTKEY_API_KEY is not set in .env."
        )
    return ModelConfig(
        model=display,
        provider="portkey",
        api_key=key,
        api_model=wire_model,
        base_url=os.getenv("PORTKEY_BASE_URL") or _portkey_settings().get("base_url") or "https://api.portkey.ai/v1",
        extra_headers=extra_headers or {},
    )


def resolve_judge(name: str) -> ModelConfig:
    """Turn a --judge argument into a ModelConfig, exiting with a clear message on failure."""
    # 2. Portkey model-catalog address
    if name.startswith("@"):
        return _portkey_config(display=name, wire_model=name)

    # 3. Portkey with header-selected provider
    if name.startswith("portkey:"):
        wire = name[len("portkey:"):]
        headers = {}
        if os.getenv("PORTKEY_PROVIDER"):
            headers["x-portkey-provider"] = os.environ["PORTKEY_PROVIDER"]
        return _portkey_config(display=name, wire_model=wire, extra_headers=headers)

    # 1. Registered model
    reg = registry()
    if name not in reg:
        sys.exit(
            f"Unknown judge '{name}'. Known judges (config/rift.yaml judges.registry): {list(reg)}. "
            f"To use any model exposed by your Portkey workspace, pass its catalog "
            f"address instead, e.g. --judge @openai/{name}"
        )
    provider, env_var = reg[name]["provider"], reg[name]["key_env"]
    direct_key = os.getenv(env_var)
    if direct_key:
        return ModelConfig(model=name, provider=provider, api_key=direct_key)

    if _portkey_available():
        slug = os.getenv(_SLUG_ENV.get(provider, "")) or (_portkey_settings().get("slugs") or {}).get(provider) or f"@{provider}"
        cfg = _portkey_config(display=name, wire_model=f"{slug}/{name}")
        print(f"  [judges] {env_var} not set; routing {name} through Portkey as {cfg.api_model}")
        return cfg

    sys.exit(
        f"No credentials for judge '{name}': set {env_var} in .env for direct access, "
        f"or set PORTKEY_API_KEY (and optionally PORTKEY_BASE_URL) to route through Portkey."
    )


def build_configs(judges: list[str]) -> list[ModelConfig]:
    return [resolve_judge(j) for j in judges]


def describe(config: ModelConfig) -> str:
    """One-line description of how a judge will be called, for run logs."""
    if config.provider == "portkey":
        return f"{config.model}  (Portkey {config.base_url} -> model={config.wire_model})"
    return f"{config.model}  (direct {config.provider})"
