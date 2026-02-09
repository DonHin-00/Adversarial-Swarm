"""
Unit Tests for Security Core Module

Tests for secure configuration and audit logging.
"""

import pytest
import os
from pathlib import Path

from src.security_core.config import SecureConfig, Environment, SecurityConfig
from src.security_core.audit import AuditLogger, AuditEvent, AuditEventType, AuditSeverity


class TestSecureConfig:
    """Test cases for SecureConfig class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        config = SecureConfig(environment=Environment.DEVELOPMENT)
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.security, SecurityConfig)
    
    def test_from_env(self):
        """Test creation from environment variables."""
        os.environ["ADVERSARIAL_SWARM_ENV"] = "production"
        config = SecureConfig.from_env()
        assert config.environment == Environment.PRODUCTION
    
    def test_get_with_default(self):
        """Test get method with default value."""
        config = SecureConfig()
        value = config.get("nonexistent_key", default="default_value")
        assert value == "default_value"
    
    def test_get_required_missing(self):
        """Test get method with required key missing."""
        config = SecureConfig()
        with pytest.raises(KeyError):
            config.get("required_key", required=True)
    
    def test_is_production(self):
        """Test production environment check."""
        config = SecureConfig(environment=Environment.PRODUCTION)
        assert config.is_production() is True
        assert config.is_development() is False
    
    def test_is_development(self):
        """Test development environment check."""
        config = SecureConfig(environment=Environment.DEVELOPMENT)
        assert config.is_development() is True
        assert config.is_production() is False


class TestAuditLogger:
    """Test cases for AuditLogger class."""
    
    def test_initialization(self):
        """Test audit logger initialization."""
        logger = AuditLogger()
        assert logger is not None
        assert logger.enable_chain is True
    
    def test_log_authentication_success(self):
        """Test logging successful authentication."""
        logger = AuditLogger()
        # Should not raise exception
        logger.log_authentication("user123", success=True, method="password")
    
    def test_log_authentication_failure(self):
        """Test logging failed authentication."""
        logger = AuditLogger()
        # Should not raise exception
        logger.log_authentication("user123", success=False, method="password")
    
    def test_log_data_access(self):
        """Test logging data access."""
        logger = AuditLogger()
        logger.log_data_access(
            user_id="user123",
            resource="customer_records",
            action="read"
        )
    
    def test_log_security_alert(self):
        """Test logging security alert."""
        logger = AuditLogger()
        logger.log_security_alert(
            alert_type="suspicious_activity",
            severity=AuditSeverity.WARNING,
            details="Multiple failed login attempts"
        )
    
    def test_audit_event_to_dict(self):
        """Test AuditEvent conversion to dictionary."""
        event = AuditEvent(
            timestamp="2026-02-09T00:00:00",
            event_type="authentication",
            severity="info",
            user_id="user123",
            action="login",
            resource=None,
            result="success",
            details={}
        )
        
        event_dict = event.to_dict()
        assert event_dict["user_id"] == "user123"
        assert event_dict["action"] == "login"
        assert event_dict["result"] == "success"
    
    def test_audit_event_compute_hash(self):
        """Test audit event hash computation."""
        event = AuditEvent(
            timestamp="2026-02-09T00:00:00",
            event_type="authentication",
            severity="info",
            user_id="user123",
            action="login",
            resource=None,
            result="success",
            details={}
        )
        
        hash1 = event.compute_hash()
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string
        
        # Same event should produce same hash
        hash2 = event.compute_hash()
        assert hash1 == hash2
        
        # Hash with previous should be different
        hash3 = event.compute_hash(previous_hash="previous")
        assert hash3 != hash1


class TestAuditDecorator:
    """Test cases for audit logging decorator."""
    
    def test_audit_log_decorator_success(self):
        """Test audit log decorator on successful function."""
        from src.security_core.audit import audit_log
        
        @audit_log(action="test_function", sensitivity="low")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_audit_log_decorator_exception(self):
        """Test audit log decorator on function that raises exception."""
        from src.security_core.audit import audit_log
        
        @audit_log(action="test_function_error", sensitivity="high")
        def test_func_error():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func_error()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
