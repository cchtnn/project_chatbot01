"""
LLM Factory - Centralized LLM creation with GROQ key rotation.
Used by ALL tools, agents, and pipelines.
"""

import os
import logging
from typing import Any, List, Optional
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from config import get_settings

logger = logging.getLogger(__name__)

# ============================================================================
# GROQ KEY ROTATION - Global State
# ============================================================================

_groq_keys_cache: List[str] = []
_groq_key_index: int = 0

def _load_groq_keys() -> List[str]:
    """Load all available GROQ API keys from settings."""
    settings = get_settings()
    keys = []
    
    # Collect all keys from settings
    if settings.groq_api_key:
        keys.append(settings.groq_api_key)
    if hasattr(settings, 'groq_api_key_2') and settings.groq_api_key_2:
        keys.append(settings.groq_api_key_2)
    if hasattr(settings, 'groq_api_key_3') and settings.groq_api_key_3:
        keys.append(settings.groq_api_key_3)
    
    # Also check environment variables directly (fallback)
    for i in range(2, 6):  # Check GROQ_API_KEY_2 through GROQ_API_KEY_5
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key and key not in keys:
            keys.append(key)
    
    return keys

def _get_next_groq_key() -> Optional[str]:
    """Round-robin through available GROQ API keys."""
    global _groq_keys_cache, _groq_key_index
    
    # Initialize cache on first call
    if not _groq_keys_cache:
        _groq_keys_cache = _load_groq_keys()
        if _groq_keys_cache:
            logger.info(f"[LLMFactory] Loaded {len(_groq_keys_cache)} GROQ API keys for rotation")
    
    if not _groq_keys_cache:
        return None
    
    # Round-robin selection
    key = _groq_keys_cache[_groq_key_index % len(_groq_keys_cache)]
    key_num = ((_groq_key_index % len(_groq_keys_cache)) + 1)
    
    logger.debug(f"[LLMFactory] Using GROQ key {key_num}/{len(_groq_keys_cache)}")
    
    _groq_key_index += 1
    return key

# ============================================================================
# MAIN LLM FACTORY
# ============================================================================

def get_llm(temperature: float = 0.1) -> Any:
    """
    Return the chat LLM used across the app with automatic GROQ key rotation.
    
    Args:
        temperature: LLM temperature (0 = deterministic, 0.1 = default)
        
    Returns:
        ChatGroq or ChatOllama instance
        
    Features:
        - Round-robin through multiple GROQ keys
        - Automatic failover if key fails
        - Falls back to Ollama if all GROQ keys fail
    """
    settings = get_settings()
    
    # Try GROQ with rotation
    if settings.llm_provider == "groq":
        groq_key = _get_next_groq_key()
        
        if groq_key:
            try:
                return ChatGroq(
                    groq_api_key=groq_key,
                    model_name=settings.groq_model,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"[LLMFactory] GROQ key failed: {e}")
                
                # Try next key if available
                if len(_groq_keys_cache) > 1:
                    logger.info("[LLMFactory] Trying next GROQ key...")
                    groq_key_backup = _get_next_groq_key()
                    
                    try:
                        return ChatGroq(
                            groq_api_key=groq_key_backup,
                            model_name=settings.groq_model,
                            temperature=temperature,
                        )
                    except Exception as e2:
                        logger.error(f"[LLMFactory] Backup GROQ key also failed: {e2}")
    
    # Fallback to Ollama
    logger.info("[LLMFactory] Using Ollama (fallback)")
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
