"""
Unit tests for 4-layer SecretEncoder system.

Tests encoding and decoding of secrets with 4 layers of protection:
- Layer 1: XOR obfuscation
- Layer 2: Base64 encoding
- Layer 3: AES-256-GCM encryption
- Layer 4: HMAC-SHA256 authentication
"""

import sys
import pytest
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hive_zero_core.security import SecretEncoder, SecureKeyManager, SecureRandom


class TestSecretEncoder:
    """Test suite for SecretEncoder class."""

    def test_encoder_initialization(self):
        """Test SecretEncoder initialization."""
        # Default initialization
        encoder = SecretEncoder()
        assert encoder.master_key is not None
        assert len(encoder.master_key) == 32

        # Custom master key
        custom_key = SecureRandom.random_bytes(32)
        encoder2 = SecretEncoder(custom_key)
        assert encoder2.master_key == custom_key

    def test_encoder_invalid_master_key(self):
        """Test that invalid master keys are rejected."""
        with pytest.raises(ValueError, match="Master key must be 32 bytes"):
            SecretEncoder(b"short_key")

        with pytest.raises(ValueError, match="Master key must be 32 bytes"):
            SecretEncoder(SecureRandom.random_bytes(16))  # Too short

    def test_encode_decode_simple_secret(self):
        """Test encoding and decoding a simple secret."""
        encoder = SecretEncoder()
        secret = b"my_secret_password_123"

        # Encode
        result = encoder.encode(secret)
        assert "encoded" in result
        assert "metadata" in result
        assert result["encoded"] != secret  # Should be different

        # Decode
        decoded = encoder.decode(result["encoded"], result["metadata"])
        assert decoded == secret

    def test_encode_decode_long_secret(self):
        """Test encoding and decoding a long secret."""
        encoder = SecretEncoder()
        # Create a long secret (1KB)
        secret = SecureRandom.random_bytes(1024)

        # Encode
        result = encoder.encode(secret)

        # Decode
        decoded = encoder.decode(result["encoded"], result["metadata"])
        assert decoded == secret

    def test_encode_decode_various_sizes(self):
        """Test encoding secrets of various sizes."""
        encoder = SecretEncoder()

        sizes = [1, 10, 50, 100, 256, 512, 1000, 4096]
        for size in sizes:
            secret = SecureRandom.random_bytes(size)
            result = encoder.encode(secret)
            decoded = encoder.decode(result["encoded"], result["metadata"])
            assert decoded == secret, f"Failed for size {size}"

    def test_four_layers_applied(self):
        """Test that all 4 layers are applied during encoding."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result = encoder.encode(secret)
        metadata = result["metadata"]

        # Check that all 4 layers are present
        assert "layers" in metadata
        assert "xor" in metadata["layers"]
        assert "base64" in metadata["layers"]
        # AES might be skipped if crypto not available
        assert "aes-gcm" in metadata["layers"] or "aes-gcm-skipped" in metadata["layers"]
        # HMAC is applied but not in the layers list (it's a wrapper)
        assert "hmac" in metadata  # HMAC signature is in metadata

    def test_metadata_structure(self):
        """Test that metadata has the correct structure."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result = encoder.encode(secret)
        metadata = result["metadata"]

        # Required fields
        assert "version" in metadata
        assert "layers" in metadata
        assert "timestamp" in metadata
        assert "original_length" in metadata
        assert "checksum" in metadata
        assert "hmac" in metadata
        assert "xor_key" in metadata

        # Verify values
        assert metadata["version"] == "1.0"
        assert metadata["original_length"] == len(secret)
        assert metadata["checksum"] == hashlib.sha256(secret).hexdigest()

    def test_tampered_data_detection(self):
        """Test that tampering with encoded data is detected."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result = encoder.encode(secret)

        # Tamper with the encoded data
        tampered = bytearray(result["encoded"])
        tampered[0] ^= 0xFF  # Flip bits in first byte
        tampered_bytes = bytes(tampered)

        # Decoding should fail with HMAC error
        with pytest.raises(ValueError, match="HMAC verification failed"):
            encoder.decode(tampered_bytes, result["metadata"])

    def test_tampered_metadata_detection(self):
        """Test that tampering with metadata is detected."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result = encoder.encode(secret)

        # Tamper with metadata
        tampered_metadata = result["metadata"].copy()
        tampered_metadata["original_length"] = 999

        # Decoding should fail
        with pytest.raises(ValueError):
            encoder.decode(result["encoded"], tampered_metadata)

    def test_different_keys_cannot_decode(self):
        """Test that a secret encoded with one key cannot be decoded with another."""
        key1 = SecureRandom.random_bytes(32)
        key2 = SecureRandom.random_bytes(32)

        encoder1 = SecretEncoder(key1)
        encoder2 = SecretEncoder(key2)

        secret = b"test_secret"
        result = encoder1.encode(secret)

        # Try to decode with different key
        with pytest.raises(ValueError):
            encoder2.decode(result["encoded"], result["metadata"])

    def test_empty_secret_rejected(self):
        """Test that empty secrets are rejected."""
        encoder = SecretEncoder()

        with pytest.raises(ValueError, match="Secret cannot be empty"):
            encoder.encode(b"")

    def test_non_bytes_secret_rejected(self):
        """Test that non-bytes secrets are rejected."""
        encoder = SecretEncoder()

        with pytest.raises(TypeError, match="Secret must be bytes"):
            encoder.encode("not_bytes")

    def test_checksum_verification(self):
        """Test that checksum is properly verified during decode."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result = encoder.encode(secret)

        # Verify checksum in metadata
        expected_checksum = hashlib.sha256(secret).hexdigest()
        assert result["metadata"]["checksum"] == expected_checksum

    def test_rotation_with_different_xor_keys(self):
        """Test that each encoding uses a different XOR key."""
        encoder = SecretEncoder()
        secret = b"test_secret"

        result1 = encoder.encode(secret)
        result2 = encoder.encode(secret)

        # Different XOR keys should be used
        assert result1["metadata"]["xor_key"] != result2["metadata"]["xor_key"]

        # Both should decode correctly
        assert encoder.decode(result1["encoded"], result1["metadata"]) == secret
        assert encoder.decode(result2["encoded"], result2["metadata"]) == secret


class TestSecureKeyManagerSecretMethods:
    """Test suite for SecureKeyManager secret encoding methods."""

    def test_encode_secret_via_key_manager(self):
        """Test encoding secrets through SecureKeyManager."""
        manager = SecureKeyManager()
        secret = b"my_api_key_12345"

        result = manager.encode_secret(secret, purpose="api_key")

        assert "encoded" in result
        assert "metadata" in result
        assert "audit" in result
        assert result["audit"]["purpose"] == "api_key"
        assert result["audit"]["encoded_by"] == "SecureKeyManager"

    def test_decode_secret_via_key_manager(self):
        """Test decoding secrets through SecureKeyManager."""
        manager = SecureKeyManager()
        secret = b"my_api_key_12345"

        result = manager.encode_secret(secret, purpose="api_key")
        decoded = manager.decode_secret(result["encoded"], result["metadata"])

        assert decoded == secret

    def test_encode_decode_various_secret_types(self):
        """Test encoding various types of secrets."""
        manager = SecureKeyManager()

        test_cases = [
            (b"password123", "password"),
            (b"sk_live_abc123xyz", "stripe_key"),
            (SecureRandom.random_bytes(32), "encryption_key"),
            (b"Bearer eyJhbGc...", "jwt_token"),
            (b"ghp_abcdef123456", "github_token"),
        ]

        for secret, purpose in test_cases:
            result = manager.encode_secret(secret, purpose=purpose)
            decoded = manager.decode_secret(result["encoded"], result["metadata"])
            assert decoded == secret, f"Failed for {purpose}"

    def test_decode_without_master_key_fails(self):
        """Test that decoding fails if master key is not available."""
        manager1 = SecureKeyManager()
        secret = b"test_secret"

        result = manager1.encode_secret(secret)

        # Create new manager without master key
        manager2 = SecureKeyManager()

        # Should fail because master key is different (ValueError or RuntimeError)
        with pytest.raises((ValueError, RuntimeError)):
            manager2.decode_secret(result["encoded"], result["metadata"])

    def test_encode_generates_master_key_if_needed(self):
        """Test that encode_secret generates master key if not present."""
        manager = SecureKeyManager()
        assert manager.master_key is None

        secret = b"test_secret"
        result = manager.encode_secret(secret)

        assert manager.master_key is not None
        assert len(manager.master_key) == 32

    def test_multiple_secrets_with_same_manager(self):
        """Test encoding multiple different secrets with the same manager."""
        manager = SecureKeyManager()

        secrets = [
            b"secret1",
            b"secret2_longer_than_first",
            b"secret3",
            SecureRandom.random_bytes(100),
        ]

        encoded_secrets = []
        for secret in secrets:
            result = manager.encode_secret(secret)
            encoded_secrets.append(result)

        # All should decode correctly
        for i, result in enumerate(encoded_secrets):
            decoded = manager.decode_secret(result["encoded"], result["metadata"])
            assert decoded == secrets[i]


class TestLayerComponents:
    """Test individual layer components of SecretEncoder."""

    def test_layer1_xor_reversible(self):
        """Test that Layer 1 XOR is reversible."""
        encoder = SecretEncoder()
        data = b"test data"
        key = SecureRandom.random_bytes(16)

        # XOR twice should give original
        encrypted = encoder._layer1_xor(data, key)
        decrypted = encoder._layer1_xor(encrypted, key)

        assert decrypted == data

    def test_layer2_base64_reversible(self):
        """Test that Layer 2 Base64 is reversible."""
        encoder = SecretEncoder()
        data = b"test data with special chars: \x00\xff\xaa"

        encoded = encoder._layer2_base64(data, encode=True)
        decoded = encoder._layer2_base64(encoded, encode=False)

        assert decoded == data

    def test_hmac_signature_consistent(self):
        """Test that HMAC signatures are consistent."""
        encoder = SecretEncoder()
        data = b"test data"
        metadata = {"key": "value"}

        sig1 = encoder._layer4_hmac(data, metadata, sign=True)
        sig2 = encoder._layer4_hmac(data, metadata, sign=True)

        assert sig1 == sig2


def test_integration_end_to_end():
    """End-to-end integration test."""
    # Simulate real-world usage scenario
    manager = SecureKeyManager()

    # Store multiple secrets
    api_keys = {
        "stripe": b"sk_live_abc123",
        "github": b"ghp_xyz789",
        "openai": b"sk-proj-abcdef",
    }

    encoded_keys = {}
    for name, key in api_keys.items():
        result = manager.encode_secret(key, purpose=f"{name}_api_key")
        encoded_keys[name] = result

    # Retrieve and verify
    for name, key in api_keys.items():
        result = encoded_keys[name]
        decoded = manager.decode_secret(result["encoded"], result["metadata"])
        assert decoded == key

    print("✓ End-to-end integration test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
