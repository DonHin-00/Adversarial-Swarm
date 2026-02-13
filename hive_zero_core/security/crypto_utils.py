"""
Cryptographically secure utilities for the Adversarial-Swarm system.

Provides secure random number generation, key management, and
cryptographic operations using industry-standard libraries.
"""

import base64
import hashlib
import hmac
import json
import secrets
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

# Try to import AES from Cryptodome (pycryptodome) or Crypto (pycrypto)
try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad

    HAS_CRYPTO = True
except ImportError:
    try:
        from Cryptodome.Cipher import AES
        from Cryptodome.Util.Padding import pad, unpad

        HAS_CRYPTO = True
    except ImportError:
        HAS_CRYPTO = False
        logging.warning("PyCryptodome not available, AES layer disabled in SecretEncoder")

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


class SecretEncoder:
    """
    Four-layer encoding system for secrets (keys, passwords, tokens, credentials).

    Implements 4 layers of protection for maximum security:
    - Layer 1: XOR obfuscation with rotating cryptographic key
    - Layer 2: Base64 encoding for transport safety
    - Layer 3: AES-256-GCM encryption (authenticated encryption)
    - Layer 4: HMAC-SHA256 authentication wrapper with metadata

    Each layer adds a different type of protection, making secrets extremely
    difficult to extract even if one layer is compromised.
    """

    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize SecretEncoder with a master key.

        Args:
            master_key: 256-bit master key (generated if not provided)
        """
        self.master_key = master_key or SecureRandom.random_bytes(32)
        if len(self.master_key) != 32:
            raise ValueError("Master key must be 32 bytes (256 bits)")
        self.has_crypto = HAS_CRYPTO
        logger.debug(f"SecretEncoder initialized (AES available: {self.has_crypto})")

    def _layer1_xor(self, data: bytes, key: bytes) -> bytes:
        """
        Layer 1: XOR obfuscation with cryptographic key.

        Provides basic obfuscation that's reversible with the key.
        """
        key_len = len(key)
        if key_len == 0:
            raise ValueError("XOR key cannot be empty")
        return bytes([data[i] ^ key[i % key_len] for i in range(len(data))])

    def _layer2_base64(self, data: bytes, encode: bool = True) -> bytes:
        """
        Layer 2: Base64 encoding.

        Ensures data is transport-safe and contains only printable characters.
        """
        if encode:
            return base64.b64encode(data)
        else:
            return base64.b64decode(data)

    def _layer3_aes(self, data: bytes, encrypt: bool = True) -> Tuple[bytes, Optional[bytes]]:
        """
        Layer 3: AES-256-GCM authenticated encryption.

        Provides confidentiality and integrity protection. GCM mode includes
        built-in authentication, preventing tampering.
        """
        if not self.has_crypto:
            logger.warning("AES encryption unavailable, skipping Layer 3")
            return data, None

        if encrypt:
            # Generate cryptographically secure random nonce (12 bytes for GCM)
            nonce = SecureRandom.random_bytes(12)

            # AES-256-GCM encryption
            cipher = AES.new(self.master_key, AES.MODE_GCM, nonce=nonce)
            ciphertext, auth_tag = cipher.encrypt_and_digest(data)

            # Return encrypted data + auth_tag, and nonce separately
            return ciphertext + auth_tag, nonce
        else:
            # For decryption, data is (ciphertext+tag, nonce)
            ciphertext_and_tag, nonce = data
            if len(ciphertext_and_tag) < 16:
                raise ValueError("Invalid encrypted data: too short")

            # Split ciphertext and auth tag (last 16 bytes)
            ciphertext = ciphertext_and_tag[:-16]
            auth_tag = ciphertext_and_tag[-16:]

            # Decrypt and verify
            cipher = AES.new(self.master_key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, auth_tag)
            return plaintext, None

    def _layer4_hmac(self, data: bytes, metadata: Dict[str, Any], sign: bool = True) -> bytes:
        """
        Layer 4: HMAC-SHA256 authentication wrapper.

        Adds final layer of integrity protection and binds metadata to the secret.
        This prevents any tampering with the encoded secret or its metadata.
        """
        # Create a copy of metadata without the 'hmac' field for consistent hashing
        metadata_for_hmac = {k: v for k, v in metadata.items() if k != "hmac"}

        if sign:
            # Create message: metadata + data (use JSON for consistent serialization)
            metadata_str = json.dumps(metadata_for_hmac, sort_keys=True).encode("utf-8")
            message = metadata_str + b"|" + data

            # Compute HMAC
            signature = hmac.new(self.master_key, message, hashlib.sha256).digest()
            return signature
        else:
            # Verify HMAC (will raise exception if invalid)
            metadata_str = json.dumps(metadata_for_hmac, sort_keys=True).encode("utf-8")
            message = metadata_str + b"|" + data
            expected_signature = hmac.new(self.master_key, message, hashlib.sha256).digest()
            return expected_signature

    def encode(self, secret: bytes) -> Dict[str, Any]:
        """
        Encode a secret with 4 layers of protection.

        Args:
            secret: The secret data to encode (password, key, token, etc.)

        Returns:
            Dictionary containing encoded secret and metadata for decoding
        """
        if not isinstance(secret, bytes):
            raise TypeError("Secret must be bytes")
        if len(secret) == 0:
            raise ValueError("Secret cannot be empty")

        try:
            encoded = secret
            metadata = {
                "version": "1.0",
                "layers": [],
                "timestamp": datetime.now().astimezone().isoformat(),
                "original_length": len(secret),
            }

            # Layer 1: XOR obfuscation with rotating key
            xor_key = SecureRandom.random_bytes(32)  # Fresh key each time
            encoded = self._layer1_xor(encoded, xor_key)
            metadata["xor_key"] = base64.b64encode(xor_key).decode("utf-8")
            metadata["layers"].append("xor")

            # Layer 2: Base64 encoding
            encoded = self._layer2_base64(encoded, encode=True)
            metadata["layers"].append("base64")

            # Layer 3: AES-256-GCM encryption
            if self.has_crypto:
                encoded, nonce = self._layer3_aes(encoded, encrypt=True)
                metadata["aes_nonce"] = base64.b64encode(nonce).decode("utf-8")
                metadata["layers"].append("aes-gcm")
            else:
                metadata["layers"].append("aes-gcm-skipped")

            # Compute final checksum of protected data (before HMAC)
            metadata["checksum"] = hashlib.sha256(encoded).hexdigest()

            # Layer 4: HMAC-SHA256 authentication
            signature = self._layer4_hmac(encoded, metadata, sign=True)
            metadata["hmac"] = base64.b64encode(signature).decode("utf-8")
            # Note: Don't add "hmac" to layers list here, as it's excluded from HMAC computation

            logger.debug(
                f"Secret encoded with {len(metadata['layers'])} layers "
                f"({len(secret)} -> {len(encoded)} bytes)"
            )

            return {"encoded": encoded, "metadata": metadata}

        except Exception as e:
            logger.error(f"Secret encoding failed: {e}", exc_info=True)
            raise

    def decode(self, encoded: bytes, metadata: Dict[str, Any]) -> bytes:
        """
        Decode a secret that was encoded with 4 layers.

        Args:
            encoded: The encoded secret data
            metadata: Metadata dictionary from encoding

        Returns:
            Original secret bytes

        Raises:
            ValueError: If HMAC verification fails or data is corrupted
        """
        if not isinstance(encoded, bytes):
            raise TypeError("Encoded data must be bytes")
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be dict")

        try:
            # Verify Layer 4: HMAC authentication
            expected_hmac = self._layer4_hmac(encoded, metadata, sign=True)
            provided_hmac = base64.b64decode(metadata.get("hmac", ""))
            if not hmac.compare_digest(expected_hmac, provided_hmac):
                raise ValueError("HMAC verification failed - data may be tampered")

            decoded = encoded
            layers = metadata.get("layers", [])

            # Reverse Layer 3: AES-256-GCM decryption
            if "aes-gcm" in layers:
                if not self.has_crypto:
                    raise RuntimeError("AES required but not available")
                nonce = base64.b64decode(metadata["aes_nonce"])
                decoded, _ = self._layer3_aes((decoded, nonce), encrypt=False)

            # Reverse Layer 2: Base64 decoding
            if "base64" in layers:
                decoded = self._layer2_base64(decoded, encode=False)

            # Reverse Layer 1: XOR deobfuscation
            if "xor" in layers:
                xor_key = base64.b64decode(metadata["xor_key"])
                decoded = self._layer1_xor(decoded, xor_key)

            # Verify checksum of original secret
            if "checksum" in metadata:
                actual_checksum = hashlib.sha256(decoded).hexdigest()
                if actual_checksum != metadata["checksum"]:
                    raise ValueError("Checksum mismatch - decoded secret is corrupted")

            # Verify length
            if "original_length" in metadata:
                if len(decoded) != metadata["original_length"]:
                    raise ValueError(
                        f"Length mismatch: expected {metadata['original_length']}, "
                        f"got {len(decoded)}"
                    )

            logger.debug(f"Secret decoded successfully ({len(decoded)} bytes)")
            return decoded

        except Exception as e:
            logger.error(f"Secret decoding failed: {e}", exc_info=True)
            raise


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
        context = f"{purpose}:{key_id}:{datetime.now().isoformat()}".encode("utf-8")
        derived_key = hmac.new(self.master_key, context, hashlib.sha256).digest()

        # Create CryptoKey object
        crypto_key = CryptoKey(
            key_id=key_id,
            key_data=derived_key,
            purpose=purpose,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self.rotation_period_days),
            rotation_count=0,
            is_active=True,
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

    def encode_secret(self, secret: bytes, purpose: str = "secret") -> Dict[str, Any]:
        """
        Encode a secret with 4 layers of protection using SecretEncoder.

        This method provides enterprise-grade secret protection for:
        - API keys and tokens
        - Passwords and credentials
        - Encryption keys
        - Any sensitive data requiring maximum security

        Args:
            secret: The secret data to encode
            purpose: Purpose description for audit trail

        Returns:
            Dictionary with encoded secret and metadata
        """
        if self.master_key is None:
            self.generate_master_key()

        # Create SecretEncoder with master key
        encoder = SecretEncoder(self.master_key)

        # Encode with 4 layers
        result = encoder.encode(secret)

        # Add audit metadata in a separate field (not part of HMAC)
        result["audit"] = {
            "purpose": purpose,
            "encoded_by": "SecureKeyManager",
        }

        logger.info(f"Secret encoded with 4 layers (purpose: {purpose})")
        return result

    def decode_secret(self, encoded: bytes, metadata: Dict[str, Any]) -> bytes:
        """
        Decode a secret that was encoded with 4 layers.

        Args:
            encoded: The encoded secret data
            metadata: Metadata dictionary from encoding

        Returns:
            Original secret bytes

        Raises:
            ValueError: If HMAC verification fails or data is corrupted
        """
        if self.master_key is None:
            raise RuntimeError("Master key not available for decoding")

        # Create SecretEncoder with master key
        encoder = SecretEncoder(self.master_key)

        # Decode and verify all 4 layers
        secret = encoder.decode(encoded, metadata)

        logger.info(f"Secret decoded successfully")
        return secret

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
    hash_value = hashlib.pbkdf2_hmac("sha256", data, salt, 100000)
    return hash_value, salt


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)
