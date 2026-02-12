"""
Cryptographically secure utilities for the Adversarial-Swarm system.

Provides secure random number generation, key management, and
cryptographic operations using industry-standard libraries.
"""

import hashlib
import hmac
import secrets
import os
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SecureRandom:
    """
    Cryptographically secure random number generator.
    
    Replaces standard `random` module for security-critical operations like:
    - ID generation
    - Token creation
    - Seed selection
    - Key generation
    """
    
    @staticmethod
    def random_int(min_val: int, max_val: int) -> int:
        """Generate cryptographically secure random integer in range [min_val, max_val]."""
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")
        return secrets.randbelow(max_val - min_val + 1) + min_val
    
    @staticmethod
    def random_float() -> float:
        """Generate cryptographically secure random float in range [0.0, 1.0)."""
        return secrets.randbelow(2**32) / (2**32)
    
    @staticmethod
    def random_bytes(n: int) -> bytes:
        """Generate n cryptographically secure random bytes."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return secrets.token_bytes(n)
    
    @staticmethod
    def random_hex(n: int) -> str:
        """Generate cryptographically secure random hex string of length n."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return secrets.token_hex(n // 2)
    
    @staticmethod
    def random_id(length: int = 12) -> str:
        """Generate cryptographically secure random ID."""
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        # Use URL-safe base64 encoding
        return secrets.token_urlsafe(length)[:length]
    
    @staticmethod
    def random_choice(seq: list):
        """Securely choose random element from sequence."""
        if not seq:
            raise ValueError("Cannot choose from empty sequence")
        return seq[secrets.randbelow(len(seq))]
    
    @staticmethod
    def random_sample(population: list, k: int) -> list:
        """Securely sample k elements from population without replacement."""
        if k > len(population):
            raise ValueError(f"Sample size ({k}) larger than population ({len(population)})")
        if k <= 0:
            raise ValueError(f"Sample size must be positive, got {k}")
        
        # Fisher-Yates shuffle with secure random
        population = list(population)  # Copy
        result = []
        for _ in range(k):
            idx = secrets.randbelow(len(population))
            result.append(population.pop(idx))
        return result


@dataclass
class CryptoKey:
    """Represents a cryptographic key with metadata."""
    key_id: str
    key_data: bytes
    purpose: str  # e.g., "encryption", "signing", "derivation"
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotation_count: int = 0
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at
    
    def should_rotate(self, rotation_period_days: int = 90) -> bool:
        """Check if key should be rotated based on age."""
        age = datetime.now() - self.created_at
        return age.days >= rotation_period_days


class SecureKeyManager:
    """
    Secure key management with rotation and versioning.
    
    Features:
    - Automatic key rotation
    - Key versioning for backward compatibility
    - Secure key storage (in-memory only, never to disk)
    - HMAC-based key derivation
    """
    
    def __init__(self, rotation_period_days: int = 90):
        self.keys: dict = {}  # key_id -> list of versions
        self.rotation_period_days = rotation_period_days
        self.master_key: Optional[bytes] = None
        logger.info(f"SecureKeyManager initialized (rotation period: {rotation_period_days} days)")
    
    def generate_master_key(self, force: bool = False) -> bytes:
        """Generate or return existing master key."""
        if self.master_key is None or force:
            self.master_key = SecureRandom.random_bytes(32)  # 256-bit key
            logger.info("New master key generated")
        return self.master_key
    
    def derive_key(self, purpose: str, key_id: Optional[str] = None) -> CryptoKey:
        """
        Derive a key from master key using HMAC-based key derivation.
        
        Args:
            purpose: Purpose of the key (e.g., "encryption", "signing")
            key_id: Optional key identifier (auto-generated if not provided)
            
        Returns:
            CryptoKey object with derived key
        """
        if self.master_key is None:
            self.generate_master_key()
        
        # Generate key ID if not provided
        if key_id is None:
            key_id = f"{purpose}_{SecureRandom.random_id(8)}"
        
        # Derive key using HMAC-SHA256
        context = f"{purpose}:{key_id}:{datetime.now().isoformat()}".encode('utf-8')
        derived_key = hmac.new(self.master_key, context, hashlib.sha256).digest()
        
        # Create CryptoKey object
        crypto_key = CryptoKey(
            key_id=key_id,
            key_data=derived_key,
            purpose=purpose,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self.rotation_period_days),
            rotation_count=0,
            is_active=True
        )
        
        # Store key (versioned)
        if key_id not in self.keys:
            self.keys[key_id] = []
        self.keys[key_id].append(crypto_key)
        
        logger.debug(f"Derived key for {purpose} (ID: {key_id})")
        return crypto_key
    
    def get_key(self, key_id: str, active_only: bool = True) -> Optional[CryptoKey]:
        """Retrieve most recent key by ID."""
        if key_id not in self.keys or not self.keys[key_id]:
            return None
        
        # Get most recent version
        for key in reversed(self.keys[key_id]):
            if active_only:
                if key.is_active and not key.is_expired():
                    return key
            else:
                return key
        
        return None
    
    def rotate_key(self, key_id: str) -> CryptoKey:
        """Rotate an existing key, keeping old version for decryption."""
        old_key = self.get_key(key_id, active_only=False)
        if old_key is None:
            raise ValueError(f"Key {key_id} not found")
        
        # Deactivate old key
        old_key.is_active = False
        
        # Create new key with same purpose
        new_key = self.derive_key(old_key.purpose, key_id)
        new_key.rotation_count = old_key.rotation_count + 1
        
        logger.info(f"Rotated key {key_id} (rotation #{new_key.rotation_count})")
        return new_key
    
    def rotate_all_keys(self):
        """Rotate all keys that should be rotated."""
        rotated = 0
        for key_id in list(self.keys.keys()):
            key = self.get_key(key_id)
            if key and key.should_rotate(self.rotation_period_days):
                self.rotate_key(key_id)
                rotated += 1
        
        if rotated > 0:
            logger.info(f"Rotated {rotated} keys")
        return rotated
    
    def compute_hmac(self, data: bytes, key_id: str) -> bytes:
        """Compute HMAC for data integrity verification."""
        key = self.get_key(key_id)
        if key is None:
            raise ValueError(f"Key {key_id} not found or expired")
        
        return hmac.new(key.key_data, data, hashlib.sha256).digest()
    
    def verify_hmac(self, data: bytes, hmac_value: bytes, key_id: str) -> bool:
        """Verify HMAC for data integrity."""
        expected_hmac = self.compute_hmac(data, key_id)
        return hmac.compare_digest(expected_hmac, hmac_value)
    
    def secure_wipe(self):
        """Securely wipe all keys from memory."""
        # Overwrite key data before deletion
        for key_versions in self.keys.values():
            for key in key_versions:
                # Overwrite with random data
                key.key_data = SecureRandom.random_bytes(len(key.key_data))
                key.is_active = False
        
        self.keys.clear()
        
        if self.master_key:
            self.master_key = SecureRandom.random_bytes(len(self.master_key))
            self.master_key = None
        
        logger.info("All keys securely wiped from memory")


def secure_hash(data: bytes, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Compute secure hash with salt.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = SecureRandom.random_bytes(32)
    
    # Use PBKDF2 with high iteration count for security
    hash_value = hashlib.pbkdf2_hmac('sha256', data, salt, 100000)
    return hash_value, salt


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)
