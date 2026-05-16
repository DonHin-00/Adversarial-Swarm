#!/usr/bin/env python3
"""
Simple test for stealth backpack module (standalone, no dependencies).
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the stealth backpack (no torch dependencies)
from hive_zero_core.agents.stealth_backpack import (
    QuadEncoder, StealthLevel, StealthBackpack, CollectionMode
)


def test_quad_encoding():
    """Test 4-layer encoding/decoding."""
    print("Test 1: Quad-Encoding")
    print("-" * 40)
    
    encoder = QuadEncoder()
    test_data = b"Secret data: admin:password123"
    
    for level in [StealthLevel.LOW, StealthLevel.MEDIUM, StealthLevel.HIGH, StealthLevel.MAXIMUM]:
        encoded_package = encoder.encode(test_data, stealth_level=level)
        decoded = encoder.decode(encoded_package['encoded'], encoded_package['metadata'])
        
        assert decoded == test_data, f"Decode failed for {level.name}"
        print(f"✓ {level.name}: {len(encoded_package['encoded'])} bytes, "
              f"layers={len(encoded_package['metadata']['layers'])}")
    
    print("✓ All encoding levels passed\n")


def test_mosquito_collection():
    """Test mosquito-style collection."""
    print("Test 2: Mosquito Collection")
    print("-" * 40)
    
    backpack = StealthBackpack(
        stealth_level=StealthLevel.MAXIMUM,
        collection_mode=CollectionMode.MOSQUITO
    )
    
    # Collect from multiple targets
    count1 = backpack.collect('target-1', specific_targets=['creds', 'keys'])
    count2 = backpack.collect('target-2', specific_targets=['config'])
    
    total = backpack.get_collected_count()
    
    print(f"✓ Collected from target-1: {count1} items")
    print(f"✓ Collected from target-2: {count2} items")
    print(f"✓ Total in backpack: {total} items")
    
    assert total == count1 + count2, "Collection count mismatch"
    print("✓ Collection test passed\n")


def test_exfiltration():
    """Test exfiltration."""
    print("Test 3: Exfiltration")
    print("-" * 40)
    
    backpack = StealthBackpack(
        stealth_level=StealthLevel.HIGH,
        collection_mode=CollectionMode.SURGICAL
    )
    
    # Collect data
    backpack.collect('server-a', specific_targets=['db_dump'])
    backpack.collect('server-b', specific_targets=['api_keys'])
    
    collected = backpack.get_collected_count()
    print(f"✓ Collected {collected} items")
    
    # Exfiltrate
    success = backpack.exfiltrate_all(channel='covert')
    
    print(f"✓ Exfiltration: {'success' if success else 'failed'}")
    print(f"✓ Remaining in backpack: {backpack.get_collected_count()}")
    print(f"✓ Total exfiltrated: {backpack.get_metrics().total_exfiltrated}")
    
    assert success, "Exfiltration failed"
    assert backpack.get_collected_count() == 0, "Backpack not cleared after exfil"
    print("✓ Exfiltration test passed\n")


def test_intelligence_harvest():
    """Test intelligence harvesting."""
    print("Test 4: Intelligence Harvesting")
    print("-" * 40)
    
    backpack = StealthBackpack(
        stealth_level=StealthLevel.MAXIMUM,
        collection_mode=CollectionMode.VACUUM
    )
    
    # Collect and exfiltrate multiple times
    backpack.collect('host-1', specific_targets=['data-a', 'data-b'])
    backpack.collect('host-2', mode=CollectionMode.VACUUM)
    backpack.exfiltrate_all()
    
    # Harvest intelligence
    intel = backpack.harvest_intelligence()
    
    print(f"✓ Total collected: {intel['metrics']['total_collected']}")
    print(f"✓ Total exfiltrated: {intel['metrics']['total_exfiltrated']}")
    print(f"✓ Avg stealth score: {intel['metrics']['avg_stealth_score']:.2f}")
    print(f"✓ Data types: {', '.join(intel['metrics']['data_types_collected'])}")
    
    assert intel['metrics']['total_collected'] > 0, "No data collected"
    assert intel['metrics']['avg_stealth_score'] > 0, "Invalid stealth score"
    print("✓ Intelligence harvest test passed\n")


def test_faraday_cage():
    """Test Faraday cage (steganographic wrapper)."""
    print("Test 5: Faraday Cage (Steganography)")
    print("-" * 40)
    
    encoder = QuadEncoder()
    malicious = b"DROP TABLE users;"
    
    # Encode with maximum stealth (includes steganography)
    encoded = encoder.encode(malicious, stealth_level=StealthLevel.MAXIMUM)
    stego_data = encoded['encoded']
    
    # Check that it looks like innocent data
    try:
        wrapper = json.loads(stego_data.decode('utf-8'))
        assert wrapper['type'] == 'network_metrics', "Wrong wrapper type"
        assert 'metrics' in wrapper, "Missing metrics"
        assert '_payload' in wrapper, "Missing payload"
        print(f"✓ Wrapper type: {wrapper['type']}")
        print(f"✓ Appears as: network_metrics with latency, packets, bandwidth")
    except Exception:
        print("✗ Failed to parse steganographic wrapper")
        raise
    
    # Decode and verify
    decoded = encoder.decode(stego_data, encoded['metadata'])
    assert decoded == malicious, "Decode failed"
    print(f"✓ Hidden payload successfully decoded")
    print("✓ Faraday cage test passed\n")


def main():
    print("=" * 50)
    print("🎒 STEALTH BACKPACK TEST SUITE")
    print("=" * 50)
    print()
    
    try:
        test_quad_encoding()
        test_mosquito_collection()
        test_exfiltration()
        test_intelligence_harvest()
        test_faraday_cage()
        
        print("=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
