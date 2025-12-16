"""
LLM utility functions for DeepCode project.

This module provides common LLM-related utilities to avoid circular imports
and reduce code duplication across the project.
"""

import os
import yaml
from typing import Any, Type, Dict, Tuple, Optional

# Import LLM classes
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# Cache for task model configuration to avoid repeated file reads
_task_model_cache: Optional[Dict] = None
_task_model_cache_mtime: float = 0


def get_model_for_task(
    task_name: str, config_path: str = "mcp_agent.config.yaml"
) -> Dict[str, Any]:
    """
    Get the model configuration for a specific task based on task_models config.

    This enables multi-model support where different tasks can use different models
    to optimize for cost, speed, or quality based on task complexity.

    Args:
        task_name: The task identifier (e.g., 'research_analyzer', 'code_implementation')
        config_path: Path to the main configuration file

    Returns:
        Dict containing:
        - provider: 'openai', 'google', or 'anthropic'
        - model: The model name to use
        - api_type: 'chat' or 'responses' (for OpenAI models)
        - llm_class: The appropriate LLM class for the provider
        - fallback: Boolean indicating if this is a fallback configuration
    """
    global _task_model_cache, _task_model_cache_mtime

    # Map providers to LLM classes
    provider_class_map = {
        "openai": OpenAIAugmentedLLM,
        "google": GoogleAugmentedLLM,
        "anthropic": AnthropicAugmentedLLM,
    }

    # Default fallback configuration
    default_config = {
        "provider": "openai",
        "model": None,  # Will use default from openai config
        "api_type": "responses",
        "llm_class": OpenAIAugmentedLLM,
        "fallback": True,
    }

    try:
        if not os.path.exists(config_path):
            print(f"⚙️ Config file {config_path} not found, using default model for {task_name}")
            return default_config

        # Check if cache is valid (file hasn't been modified)
        current_mtime = os.path.getmtime(config_path)
        if _task_model_cache is not None and current_mtime == _task_model_cache_mtime:
            task_models = _task_model_cache
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            task_models = config.get("task_models", {})
            _task_model_cache = task_models
            _task_model_cache_mtime = current_mtime

        # Search for the task in all tiers
        for tier_name, tier_config in task_models.items():
            if not isinstance(tier_config, dict):
                continue
            tasks = tier_config.get("tasks", [])
            if task_name in tasks:
                provider = tier_config.get("provider", "openai")
                model = tier_config.get("model")
                api_type = tier_config.get("api_type", "chat")
                llm_class = provider_class_map.get(provider, OpenAIAugmentedLLM)

                result = {
                    "provider": provider,
                    "model": model,
                    "api_type": api_type,
                    "llm_class": llm_class,
                    "fallback": False,
                    "tier": tier_name,
                }
                print(f"⚙️ Task '{task_name}' -> {tier_name} tier: {provider}/{model}")
                return result

        # Task not found in any tier, use default
        print(f"⚙️ Task '{task_name}' not configured, using default model")
        return default_config

    except Exception as e:
        print(f"⚠️ Error reading task model config for '{task_name}': {e}")
        return default_config


def get_llm_class_for_task(
    task_name: str, config_path: str = "mcp_agent.config.yaml"
) -> Type[Any]:
    """
    Convenience function to get just the LLM class for a task.

    Args:
        task_name: The task identifier
        config_path: Path to the configuration file

    Returns:
        The LLM class (OpenAIAugmentedLLM, GoogleAugmentedLLM, or AnthropicAugmentedLLM)
    """
    config = get_model_for_task(task_name, config_path)
    return config["llm_class"]


def get_preferred_llm_class(config_path: str = "mcp_agent.secrets.yaml") -> Type[Any]:
    """
    Select the LLM class based on user preference and API key availability.

    Priority:
    1. Check mcp_agent.config.yaml for llm_provider preference
    2. Verify the preferred provider has API key
    3. Fallback to first available provider

    Args:
        config_path: Path to the secrets YAML configuration file

    Returns:
        class: The preferred LLM class
    """
    try:
        # Read API keys from secrets file
        if not os.path.exists(config_path):
            print(f"🤖 Config file {config_path} not found, using OpenAIAugmentedLLM")
            return OpenAIAugmentedLLM

        with open(config_path, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)

        # Get API keys
        anthropic_key = secrets.get("anthropic", {}).get("api_key", "").strip()
        google_key = secrets.get("google", {}).get("api_key", "").strip()
        openai_key = secrets.get("openai", {}).get("api_key", "").strip()

        # Read user preference from main config
        main_config_path = "mcp_agent.config.yaml"
        preferred_provider = None
        if os.path.exists(main_config_path):
            with open(main_config_path, "r", encoding="utf-8") as f:
                main_config = yaml.safe_load(f)
                preferred_provider = main_config.get("llm_provider", "").strip().lower()

        # Map of providers to their classes and keys
        provider_map = {
            "anthropic": (
                AnthropicAugmentedLLM,
                anthropic_key,
                "AnthropicAugmentedLLM",
            ),
            "google": (GoogleAugmentedLLM, google_key, "GoogleAugmentedLLM"),
            "openai": (OpenAIAugmentedLLM, openai_key, "OpenAIAugmentedLLM"),
        }

        # Try user's preferred provider first
        if preferred_provider and preferred_provider in provider_map:
            llm_class, api_key, class_name = provider_map[preferred_provider]
            if api_key:
                print(f"🤖 Using {class_name} (user preference: {preferred_provider})")
                return llm_class
            else:
                print(
                    f"⚠️ Preferred provider '{preferred_provider}' has no API key, checking alternatives..."
                )

        # Fallback: try providers in order of availability
        for provider, (llm_class, api_key, class_name) in provider_map.items():
            if api_key:
                print(f"🤖 Using {class_name} ({provider} API key found)")
                return llm_class

        # No API keys found
        print("⚠️ No API keys configured, falling back to OpenAIAugmentedLLM")
        return OpenAIAugmentedLLM

    except Exception as e:
        print(f"🤖 Error reading config file {config_path}: {e}")
        print("🤖 Falling back to OpenAIAugmentedLLM")
        return OpenAIAugmentedLLM


def get_token_limits(config_path: str = "mcp_agent.config.yaml") -> Tuple[int, int]:
    """
    Get token limits from configuration.

    Args:
        config_path: Path to the main configuration file

    Returns:
        tuple: (base_max_tokens, retry_max_tokens)
    """
    # Default values that work with qwen/qwen-max (32768 total context)
    default_base = 20000
    default_retry = 15000

    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            openai_config = config.get("openai", {})
            base_tokens = openai_config.get("base_max_tokens", default_base)
            retry_tokens = openai_config.get("retry_max_tokens", default_retry)

            print(
                f"⚙️ Token limits from config: base={base_tokens}, retry={retry_tokens}"
            )
            return base_tokens, retry_tokens
        else:
            print(
                f"⚠️ Config file {config_path} not found, using defaults: base={default_base}, retry={default_retry}"
            )
            return default_base, default_retry
    except Exception as e:
        print(f"⚠️ Error reading token config from {config_path}: {e}")
        print(
            f"🔧 Falling back to default token limits: base={default_base}, retry={default_retry}"
        )
        return default_base, default_retry


def get_default_models(config_path: str = "mcp_agent.config.yaml"):
    """
    Get default models from configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Dictionary with 'anthropic', 'openai', and 'google' default models
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Handle null values in config sections
            anthropic_config = config.get("anthropic") or {}
            openai_config = config.get("openai") or {}
            google_config = config.get("google") or {}

            anthropic_model = anthropic_config.get(
                "default_model", "claude-sonnet-4-20250514"
            )
            openai_model = openai_config.get("default_model", "o3-mini")
            google_model = google_config.get("default_model", "gemini-2.0-flash")

            return {
                "anthropic": anthropic_model,
                "openai": openai_model,
                "google": google_model,
            }
        else:
            print(f"Config file {config_path} not found, using default models")
            return {
                "anthropic": "claude-sonnet-4-20250514",
                "openai": "o3-mini",
                "google": "gemini-2.0-flash",
            }

    except Exception as e:
        print(f"❌Error reading config file {config_path}: {e}")
        return {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "o3-mini",
            "google": "gemini-2.0-flash",
        }


def get_document_segmentation_config(
    config_path: str = "mcp_agent.config.yaml",
) -> Dict[str, Any]:
    """
    Get document segmentation configuration from config file.

    Args:
        config_path: Path to the main configuration file

    Returns:
        Dict containing segmentation configuration with default values
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Get document segmentation config with defaults
            seg_config = config.get("document_segmentation", {})
            return {
                "enabled": seg_config.get("enabled", True),
                "size_threshold_chars": seg_config.get("size_threshold_chars", 50000),
            }
        else:
            print(
                f"📄 Config file {config_path} not found, using default segmentation settings"
            )
            return {"enabled": True, "size_threshold_chars": 50000}

    except Exception as e:
        print(f"📄 Error reading segmentation config from {config_path}: {e}")
        print("📄 Using default segmentation settings")
        return {"enabled": True, "size_threshold_chars": 50000}


def should_use_document_segmentation(
    document_content: str, config_path: str = "mcp_agent.config.yaml"
) -> Tuple[bool, str]:
    """
    Determine whether to use document segmentation based on configuration and document size.

    Args:
        document_content: The content of the document to analyze
        config_path: Path to the configuration file

    Returns:
        Tuple of (should_segment, reason) where:
        - should_segment: Boolean indicating whether to use segmentation
        - reason: String explaining the decision
    """
    seg_config = get_document_segmentation_config(config_path)

    if not seg_config["enabled"]:
        return False, "Document segmentation disabled in configuration"

    doc_size = len(document_content)
    threshold = seg_config["size_threshold_chars"]

    if doc_size > threshold:
        return (
            True,
            f"Document size ({doc_size:,} chars) exceeds threshold ({threshold:,} chars)",
        )
    else:
        return (
            False,
            f"Document size ({doc_size:,} chars) below threshold ({threshold:,} chars)",
        )


def get_adaptive_agent_config(
    use_segmentation: bool, search_server_names: list = None
) -> Dict[str, list]:
    """
    Get adaptive agent configuration based on whether to use document segmentation.

    Args:
        use_segmentation: Whether to include document-segmentation server
        search_server_names: Base search server names (from get_search_server_names)

    Returns:
        Dict containing server configurations for different agents
    """
    if search_server_names is None:
        search_server_names = []

    # Base configuration
    config = {
        "concept_analysis": [],
        "algorithm_analysis": search_server_names.copy(),
        "code_planner": search_server_names.copy(),
    }

    # Add document-segmentation server if needed
    if use_segmentation:
        config["concept_analysis"] = ["document-segmentation"]
        if "document-segmentation" not in config["algorithm_analysis"]:
            config["algorithm_analysis"].append("document-segmentation")
        if "document-segmentation" not in config["code_planner"]:
            config["code_planner"].append("document-segmentation")
    else:
        config["concept_analysis"] = ["filesystem"]
        if "filesystem" not in config["algorithm_analysis"]:
            config["algorithm_analysis"].append("filesystem")
        if "filesystem" not in config["code_planner"]:
            config["code_planner"].append("filesystem")

    return config


def get_adaptive_prompts(use_segmentation: bool) -> Dict[str, str]:
    """
    Get appropriate prompt versions based on segmentation usage.

    Args:
        use_segmentation: Whether to use segmented reading prompts

    Returns:
        Dict containing prompt configurations
    """
    # Import here to avoid circular imports
    from prompts.code_prompts import (
        PAPER_CONCEPT_ANALYSIS_PROMPT,
        PAPER_ALGORITHM_ANALYSIS_PROMPT,
        CODE_PLANNING_PROMPT,
        PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
        PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
        CODE_PLANNING_PROMPT_TRADITIONAL,
    )

    if use_segmentation:
        return {
            "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT,
            "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT,
            "code_planning": CODE_PLANNING_PROMPT,
        }
    else:
        return {
            "concept_analysis": PAPER_CONCEPT_ANALYSIS_PROMPT_TRADITIONAL,
            "algorithm_analysis": PAPER_ALGORITHM_ANALYSIS_PROMPT_TRADITIONAL,
            "code_planning": CODE_PLANNING_PROMPT_TRADITIONAL,
        }
