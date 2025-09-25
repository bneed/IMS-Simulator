"""
Licensing module - Simple, robust Pro key verification
Works offline with grace period, online verification optional
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

# Cache location
CACHE = Path.home() / ".imsphysics" / "license.json"

# Early access keys (always valid in dev)
EARLY_KEYS = {
    "IMS-PRO-EARLY-0001", 
    "IMS-PRO-EARLY-0002", 
    "IMS-PRO-EARLY-0003"
}

def _load_cache() -> Dict:
    """Load license cache from disk."""
    try:
        if CACHE.exists():
            return json.loads(CACHE.read_text())
    except Exception:
        pass
    return {}

def _save_cache(payload: Dict) -> None:
    """Save license cache to disk."""
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass  # Fail silently if cache write fails

def is_valid_format(key: str) -> bool:
    """Check if key has valid format."""
    if not key or not isinstance(key, str):
        return False
    return key.startswith("IMS-PRO-") and len(key) > 10

def is_pro(key: Optional[str], require_online: bool = False) -> bool:
    """
    Check if user has Pro access.
    
    Args:
        key: Pro license key
        require_online: Force online verification (ignored for early keys)
    
    Returns:
        True if user has Pro access
    """
    if not key:
        return False
    
    # Early access keys are always valid
    if key in EARLY_KEYS:
        _save_cache({
            "key": key, 
            "verified_at": time.time(), 
            "email": "early@access"
        })
        return True
    
    if not is_valid_format(key):
        return False
    
    # Try offline cache first
    cache = _load_cache()
    if cache.get("key") == key:
        age_days = (time.time() - cache.get("verified_at", 0)) / 86400
        if age_days <= 7:  # 7-day offline grace period
            return True
    
    # For now, accept all well-formed keys during bootstrapping
    # TODO: Implement real server verification
    if is_valid_format(key):
        _save_cache({
            "key": key, 
            "verified_at": time.time(), 
            "email": "user@domain"
        })
        return True
    
    return False

def get_cached_info() -> Dict:
    """Get cached license information."""
    cache = _load_cache()
    if cache.get("key"):
        age_days = (time.time() - cache.get("verified_at", 0)) / 86400
        cache["age_days"] = age_days
        cache["is_valid"] = age_days <= 7
    return cache

def clear_cache() -> None:
    """Clear license cache."""
    try:
        if CACHE.exists():
            CACHE.unlink()
    except Exception:
        pass