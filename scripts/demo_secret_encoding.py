#!/usr/bin/env python3
"""
Demonstration of the 4-Layer Secret Encoding System

This script showcases the enterprise-grade secret protection provided by
the Adversarial-Swarm security infrastructure.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hive_zero_core.security import SecretEncoder, SecureKeyManager


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_direct_encoding():
    """Demonstrate direct SecretEncoder usage."""
    print_section("Demo 1: Direct SecretEncoder Usage")

    encoder = SecretEncoder()
    secret = b"sk_live_stripe_api_key_xyz123"

    print(f"\n📝 Original Secret: {secret.decode()}")
    print(f"   Length: {len(secret)} bytes")

    # Encode
    result = encoder.encode(secret)
    print(f"\n🔒 Encoded with {len(result['metadata']['layers'])} layers:")
    for i, layer in enumerate(result["metadata"]["layers"], 1):
        print(f"   Layer {i}: {layer}")

    print(f"\n📦 Encoded Data:")
    print(f"   Size: {len(result['encoded'])} bytes")
    print(f"   Overhead: {len(result['encoded']) - len(secret)} bytes "
          f"({100 * (len(result['encoded']) - len(secret)) / len(secret):.1f}%)")

    # Decode
    decoded = encoder.decode(result["encoded"], result["metadata"])
    print(f"\n✅ Decoded Successfully: {decoded == secret}")
    print(f"   Decoded: {decoded.decode()}")


def demo_key_manager():
    """Demonstrate SecureKeyManager integration."""
    print_section("Demo 2: SecureKeyManager Integration")

    manager = SecureKeyManager()

    secrets = [
        (b"sk_live_stripe_abc123", "stripe_api_key"),
        (b"ghp_github_token_xyz789", "github_pat"),
        (b"sk-proj-openai-def456", "openai_api_key"),
        (b"postgres://user:pass@host/db", "database_url"),
    ]

    print("\n🔐 Encoding multiple secrets:")
    encoded_secrets = []
    for secret, purpose in secrets:
        result = manager.encode_secret(secret, purpose=purpose)
        encoded_secrets.append((result, secret, purpose))
        print(f"   ✓ {purpose}: {len(secret)} → {len(result['encoded'])} bytes")

    print("\n🔓 Decoding all secrets:")
    for result, original, purpose in encoded_secrets:
        decoded = manager.decode_secret(result["encoded"], result["metadata"])
        status = "✅" if decoded == original else "❌"
        print(f"   {status} {purpose}: {decoded == original}")


def demo_security_features():
    """Demonstrate security features."""
    print_section("Demo 3: Security Features")

    encoder = SecretEncoder()
    secret = b"secret_password_123"

    result = encoder.encode(secret)

    # Test 1: Tampering detection
    print("\n🔒 Test 1: Tampering Detection")
    tampered = bytearray(result["encoded"])
    tampered[0] ^= 0xFF  # Flip bits
    try:
        encoder.decode(bytes(tampered), result["metadata"])
        print("   ❌ FAIL: Tampering not detected")
    except ValueError as e:
        print("   ✅ PASS: Tampering detected")
        print(f"      Error: {e}")

    # Test 2: Different key rejection
    print("\n🔑 Test 2: Different Key Rejection")
    encoder2 = SecretEncoder()  # Different master key
    try:
        encoder2.decode(result["encoded"], result["metadata"])
        print("   ❌ FAIL: Different key allowed decode")
    except ValueError:
        print("   ✅ PASS: Different key rejected")

    # Test 3: Metadata tampering
    print("\n📝 Test 3: Metadata Tampering Detection")
    tampered_metadata = result["metadata"].copy()
    tampered_metadata["original_length"] = 999
    try:
        encoder.decode(result["encoded"], tampered_metadata)
        print("   ❌ FAIL: Metadata tampering not detected")
    except ValueError:
        print("   ✅ PASS: Metadata tampering detected")

    # Test 4: Freshness (unique encodings)
    print("\n🔄 Test 4: Freshness (Unique Encodings)")
    result1 = encoder.encode(secret)
    result2 = encoder.encode(secret)
    if result1["encoded"] != result2["encoded"]:
        print("   ✅ PASS: Same secret produces different encodings")
        print("      Prevents replay attacks")
    else:
        print("   ❌ FAIL: Same secret produces identical encodings")


def demo_performance():
    """Demonstrate performance characteristics."""
    print_section("Demo 4: Performance Characteristics")

    import time

    encoder = SecretEncoder()
    sizes = [16, 64, 256, 1024, 4096]

    print("\n⚡ Encoding Performance:")
    for size in sizes:
        secret = b"A" * size
        start = time.time()
        for _ in range(100):
            result = encoder.encode(secret)
        elapsed = (time.time() - start) / 100
        print(f"   {size:5d} bytes: {elapsed*1000:6.2f} ms/op")

    print("\n⚡ Decoding Performance:")
    results = {}
    for size in sizes:
        secret = b"A" * size
        results[size] = encoder.encode(secret)

    for size in sizes:
        result = results[size]
        start = time.time()
        for _ in range(100):
            decoded = encoder.decode(result["encoded"], result["metadata"])
        elapsed = (time.time() - start) / 100
        print(f"   {size:5d} bytes: {elapsed*1000:6.2f} ms/op")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  4-LAYER SECRET ENCODING SYSTEM DEMONSTRATION")
    print("  Adversarial-Swarm Security Infrastructure")
    print("=" * 70)

    try:
        demo_direct_encoding()
        demo_key_manager()
        demo_security_features()
        demo_performance()

        print("\n" + "=" * 70)
        print("  ✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
