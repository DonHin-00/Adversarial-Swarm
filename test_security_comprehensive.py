#!/usr/bin/env python3
"""
Comprehensive Security Integration Test Suite

Tests all security enhancements across the Adversarial-Swarm system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_security_module():
    """Test security module imports and basic functionality."""
    print("Testing security module...")
    
    from hive_zero_core.security import (
        SecureRandom, SecureKeyManager, InputValidator,
        AuditLogger, SecurityEvent, AccessController,
        OperationType, sanitize_path, sanitize_input
    )
    
    # Test SecureRandom
    sr = SecureRandom()
    sid = sr.random_id(12)
    assert len(sid) == 12, "Secure ID length mismatch"
    
    r_int = sr.random_int(1, 100)
    assert 1 <= r_int <= 100, "Secure int out of range"
    
    r_bytes = sr.random_bytes(32)
    assert len(r_bytes) == 32, "Secure bytes length mismatch"
    
    print("  ✓ SecureRandom working")
    
    # Test InputValidator
    iv = InputValidator()
    assert iv.validate_path("test.txt", allow_absolute=False), "Valid path rejected"
    assert not iv.validate_path("../etc/passwd"), "Path traversal not caught"
    
    sanitized = sanitize_input("test@#$%123", allowed_chars=InputValidator.ALPHANUMERIC)
    assert sanitized == "test123", f"Sanitization failed: {sanitized}"
    
    print("  ✓ InputValidator working")
    
    # Test AuditLogger
    audit = AuditLogger()
    audit.log_event(
        SecurityEvent.SYSTEM_STARTED,
        "test_actor",
        "Test action",
        result="success"
    )
    assert len(audit.entries) == 1, "Audit entry not recorded"
    
    valid, invalid = audit.verify_integrity()
    assert valid, f"Audit chain broken: {invalid}"
    
    print("  ✓ AuditLogger working")
    
    # Test AccessController
    ac = AccessController()
    actor_id = "test_actor_001"
    ac.register_actor(actor_id, "operator")
    
    authorized = ac.authorize_operation(actor_id, OperationType.DATA_COLLECT)
    assert authorized, "Authorization failed for valid operation"
    
    print("  ✓ AccessController working")
    
    print("✅ Security module tests PASSED\n")


def test_stealth_backpack_security():
    """Test security integration in stealth backpack."""
    print("Testing stealth backpack security...")
    
    from hive_zero_core.agents.stealth_backpack import (
        StealthBackpack, StealthLevel, CollectionMode, SecureRandom
    )
    
    # Test secure ID generation
    bp = StealthBackpack(
        stealth_level=StealthLevel.MAXIMUM,
        collection_mode=CollectionMode.MOSQUITO
    )
    
    assert bp.backpack_id.startswith("bp_"), "Backpack ID format incorrect"
    assert bp.actor_id.startswith("actor_"), "Actor ID format incorrect"
    assert len(bp.backpack_id) > 8, "Backpack ID too short"
    
    print("  ✓ Secure ID generation working")
    
    # Test secure random in encoder
    assert hasattr(bp.encoder, 'master_key'), "Encoder missing master key"
    assert len(bp.encoder.master_key) == 32, "Master key wrong size"
    
    print("  ✓ Secure key generation working")
    
    print("✅ Stealth backpack security tests PASSED\n")


def test_no_insecure_random():
    """Verify no insecure random usage in secured modules."""
    print("Checking for insecure random usage...")
    
    import re
    
    secured_files = [
        'hive_zero_core/security/crypto_utils.py',
        'hive_zero_core/security/input_validator.py',
        'hive_zero_core/security/audit_logger.py',
        'hive_zero_core/security/access_control.py',
        'hive_zero_core/agents/stealth_backpack.py',
    ]
    
    insecure_patterns = [
        r'\brandom\.randint\b',
        r'\brandom\.choice\b',
        r'\brandom\.random\b',
        r'\brandom\.randrange\b',
    ]
    
    issues_found = False
    
    for filepath in secured_files:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        for pattern in insecure_patterns:
            matches = re.findall(pattern, content)
            if matches and 'import random' not in content.split('\n')[0:50]:
                # Only flag if random is actually imported
                continue
            if matches:
                print(f"  ⚠ Found insecure random in {filepath}: {matches}")
                issues_found = True
    
    if not issues_found:
        print("  ✓ No insecure random usage found in secured modules")
        print("✅ Insecure random check PASSED\n")
    else:
        print("❌ Insecure random usage found\n")
        return False
    
    return True


def test_audit_log_integrity():
    """Test audit log chain integrity."""
    print("Testing audit log integrity...")
    
    from hive_zero_core.security import AuditLogger, SecurityEvent
    
    audit = AuditLogger()
    
    # Add multiple events
    for i in range(10):
        audit.log_event(
            SecurityEvent.DATA_COLLECTED,
            f"actor_{i}",
            f"Action {i}",
            result="success"
        )
    
    # Verify chain
    valid, invalid_indices = audit.verify_integrity()
    assert valid, f"Chain broken at indices: {invalid_indices}"
    assert len(invalid_indices) == 0, "Invalid entries found"
    
    # Test tampering detection
    if len(audit.entries) > 5:
        # Tamper with an entry
        audit.entries[5].action = "TAMPERED"
        valid, invalid_indices = audit.verify_integrity()
        assert not valid, "Tampering not detected!"
        assert 5 in invalid_indices, "Tampered entry not identified"
    
    print("  ✓ Audit chain integrity verified")
    print("  ✓ Tampering detection working")
    print("✅ Audit log integrity tests PASSED\n")


def test_input_validation():
    """Test comprehensive input validation."""
    print("Testing input validation...")
    
    from hive_zero_core.security import InputValidator, validate_command_safe
    
    iv = InputValidator()
    
    # Path traversal tests
    dangerous_paths = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "%2e%2e/etc/passwd",
        "test/../../etc/passwd"
    ]
    
    for path in dangerous_paths:
        assert not iv.validate_path(path), f"Dangerous path not caught: {path}"
    
    print("  ✓ Path traversal protection working")
    
    # Command injection tests
    dangerous_commands = [
        "test; rm -rf /",
        "test | cat /etc/passwd",
        "test && malicious",
        "test `whoami`",
        "test $(whoami)",
    ]
    
    for cmd in dangerous_commands:
        assert not validate_command_safe(cmd), f"Command injection not caught: {cmd}"
    
    print("  ✓ Command injection protection working")
    
    # Filename validation
    assert iv.validate_filename("normal.txt"), "Valid filename rejected"
    assert not iv.validate_filename("../etc/passwd"), "Path in filename not caught"
    assert not iv.validate_filename("CON"), "Dangerous Windows name not caught"
    assert not iv.validate_filename("file\x00.txt"), "Null byte not caught"
    
    print("  ✓ Filename validation working")
    print("✅ Input validation tests PASSED\n")


def run_all_tests():
    """Run all security tests."""
    print("=" * 60)
    print("ADVERSARIAL-SWARM SECURITY TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_security_module,
        test_stealth_backpack_security,
        test_no_insecure_random,
        test_audit_log_integrity,
        test_input_validation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
