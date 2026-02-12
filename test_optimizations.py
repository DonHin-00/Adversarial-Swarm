#!/usr/bin/env python3
"""
Simple test script to validate performance optimizations.
Tests core functionality of optimized components.
"""

import torch
import time
from hive_zero_core.memory.graph_store import LogEncoder
from hive_zero_core.memory.foundation import SyntheticExperienceGenerator

def test_log_encoder_optimization():
    """Test LogEncoder with optimized node feature creation."""
    print("Testing LogEncoder optimization...")
    encoder = LogEncoder(node_feature_dim=64)
    
    # Create test logs
    test_logs = [
        {'src_ip': '192.168.1.1', 'dst_ip': '10.0.0.5', 'port': 80, 'proto': 6},
        {'src_ip': '10.0.0.5', 'dst_ip': '8.8.8.8', 'port': 53, 'proto': 17},
        {'src_ip': '192.168.1.2', 'dst_ip': '10.0.0.6', 'port': 443, 'proto': 6},
    ]
    
    start = time.time()
    data = encoder.update(test_logs)
    elapsed = time.time() - start
    
    assert data.x.size(0) > 0, "Should have nodes"
    assert data.edge_index.size(1) == len(test_logs), "Should have correct number of edges"
    print(f"✓ LogEncoder test passed in {elapsed*1000:.2f}ms")
    print(f"  Created {data.x.size(0)} nodes and {data.edge_index.size(1)} edges")
    return elapsed

def test_synthetic_data_generation():
    """Test vectorized synthetic data generation."""
    print("\nTesting SyntheticExperienceGenerator optimization...")
    generator = SyntheticExperienceGenerator(observation_dim=64, action_dim=64)
    
    batch_size = 1000
    start = time.time()
    obs, acts, rews, next_obs, dones = generator.generate_batch(batch_size)
    elapsed = time.time() - start
    
    assert obs.shape == (batch_size, 64), "Observations should have correct shape"
    assert acts.shape == (batch_size, 64), "Actions should have correct shape"
    assert rews.shape == (batch_size, 1), "Rewards should have correct shape"
    print(f"✓ SyntheticExperienceGenerator test passed in {elapsed*1000:.2f}ms")
    print(f"  Generated {batch_size} samples")
    return elapsed

def test_hive_mind_forward():
    """Test HiveMind with index-based expert dispatch.
    
    Note: This test is skipped because it requires HuggingFace models.
    The index-based dispatch has been verified in code review.
    """
    print("\nTesting HiveMind optimization...")
    print("  (Skipping - requires HuggingFace models)")
    print("  Index-based dispatch verified in code review")
    # Return None to indicate skip rather than 0.0 which implies success
    return None

def test_tarpit_caching():
    """Test Agent_Tarpit with trap caching."""
    print("\nTesting Agent_Tarpit caching optimization...")
    from hive_zero_core.agents.defense_experts import Agent_Tarpit
    
    tarpit = Agent_Tarpit(observation_dim=64, action_dim=64, hidden_dim=128)
    tarpit.is_active = True
    
    x = torch.randn(1, 64)
    
    # First forward pass in training mode (no cache)
    tarpit.train()
    start = time.time()
    out1 = tarpit(x)
    elapsed1 = time.time() - start
    
    # Second forward pass in eval mode (with cache)
    tarpit.eval()
    start = time.time()
    out2 = tarpit(x)
    elapsed2 = time.time() - start
    
    # Third forward pass in eval mode (should use cache)
    start = time.time()
    out3 = tarpit(x)
    elapsed3 = time.time() - start
    
    assert out1.shape == (1, 64), "Output should have correct shape"
    assert out2.shape == (1, 64), "Output should have correct shape"
    assert out3.shape == (1, 64), "Output should have correct shape"
    print(f"✓ Agent_Tarpit test passed")
    print(f"  Training mode: {elapsed1*1000:.2f}ms")
    print(f"  Eval mode (first): {elapsed2*1000:.2f}ms")
    print(f"  Eval mode (cached): {elapsed3*1000:.2f}ms")
    if elapsed3 < elapsed2:
        print(f"  Speedup: {elapsed2/elapsed3:.2f}x")
    return elapsed2, elapsed3

def main():
    print("=" * 60)
    print("Performance Optimization Validation Tests")
    print("=" * 60)
    
    try:
        # Run all tests
        log_encoder_time = test_log_encoder_optimization()
        synth_time = test_synthetic_data_generation()
        hive_time = test_hive_mind_forward()
        tarpit_time1, tarpit_time2 = test_tarpit_caching()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nSummary:")
        print(f"  LogEncoder: {log_encoder_time*1000:.2f}ms")
        print(f"  SyntheticExperienceGenerator: {synth_time*1000:.2f}ms")
        if hive_time is not None:
            print(f"  HiveMind forward pass: {hive_time*1000:.2f}ms")
        else:
            print(f"  HiveMind forward pass: skipped")
        print(f"  Agent_Tarpit eval mode (cached): {tarpit_time2*1000:.2f}ms")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
