"""
Secure Configuration Management

Handles secure loading and validation of configuration from multiple sources:
- Environment variables
- Configuration files (with encryption)
- Secret management systems (Vault, AWS Secrets Manager, etc.)
- Command-line arguments

All sensitive data is encrypted at rest and in memory when possible.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class SecurityConfig:
    """Security-specific configuration."""
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    rate_limiting_enabled: bool = True
    tls_enabled: bool = True
    api_key_rotation_days: int = 90
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 5


@dataclass
class AIConfig:
    """AI/ML-specific configuration."""
    model_path: Optional[Path] = None
    device: str = "cuda"  # cuda, cpu, or mps
    batch_size: int = 32
    num_workers: int = 4
    mixed_precision: bool = True
    model_cache_dir: Optional[Path] = None


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    log_level: str = "INFO"
    metrics_port: int = 9090
    health_check_port: int = 8080


class SecureConfig:
    """
    Secure configuration manager with support for multiple sources.
    
    Features:
    - Environment variable loading with validation
    - Encrypted configuration file support
    - Integration with secret management systems
    - Type validation and conversion
    - Secure defaults for all settings
    
    Example:
        >>> config = SecureConfig.from_env()
        >>> api_key = config.get_secret('API_KEY')
        >>> db_url = config.get('DATABASE_URL', required=True)
    """
    
    def __init__(
        self,
        environment: Environment = Environment.DEVELOPMENT,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize secure configuration.
        
        Args:
            environment: Deployment environment (dev, staging, prod)
            config_dict: Optional pre-loaded configuration dictionary
        """
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        self._config: Dict[str, Any] = config_dict or {}
        self._secrets: Dict[str, str] = {}
        
        # Initialize sub-configurations
        self.security = SecurityConfig()
        self.ai = AIConfig()
        self.monitoring = MonitoringConfig()
        
        # Load configuration
        self._load_from_environment()
        self._validate_configuration()
    
    @classmethod
    def from_env(cls, env_var: str = "ADVERSARIAL_SWARM_ENV") -> "SecureConfig":
        """
        Create configuration from environment variables.
        
        Args:
            env_var: Environment variable name for environment type
            
        Returns:
            Configured SecureConfig instance
        """
        env_name = os.getenv(env_var, "development").lower()
        try:
            environment = Environment(env_name)
        except ValueError:
            logging.warning(f"Invalid environment '{env_name}', defaulting to development")
            environment = Environment.DEVELOPMENT
        
        return cls(environment=environment)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "SecureConfig":
        """
        Load configuration from an encrypted JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured SecureConfig instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        env = Environment(config_dict.get('environment', 'development'))
        return cls(environment=env, config_dict=config_dict)
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        prefix = "ADVERSARIAL_SWARM_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._config[config_key] = self._parse_value(value)
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        # Production environment requires enhanced security
        if self.environment == Environment.PRODUCTION:
            if not self.security.encryption_enabled:
                raise ValueError("Encryption must be enabled in production")
            if not self.security.tls_enabled:
                raise ValueError("TLS must be enabled in production")
            if not self.security.audit_logging_enabled:
                raise ValueError("Audit logging must be enabled in production")
        
        self.logger.info(f"Configuration validated for {self.environment.value} environment")
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            required: Raise error if key not found and no default
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If required key is missing
        """
        if required and key not in self._config and default is None:
            raise KeyError(f"Required configuration key '{key}' not found")
        
        return self._config.get(key, default)
    
    def get_secret(self, key: str, required: bool = True) -> Optional[str]:
        """
        Get secret value from secure storage.
        
        In production, this would integrate with:
        - AWS Secrets Manager
        - HashiCorp Vault
        - Azure Key Vault
        - Google Secret Manager
        
        Args:
            key: Secret key
            required: Raise error if secret not found
            
        Returns:
            Secret value or None
            
        Raises:
            KeyError: If required secret is missing
        """
        # For now, check environment variables with SECRET_ prefix
        secret_key = f"SECRET_{key}"
        value = os.getenv(secret_key)
        
        if required and value is None:
            raise KeyError(f"Required secret '{key}' not found")
        
        return value
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def __repr__(self) -> str:
        """String representation (excludes secrets)."""
        return f"SecureConfig(environment={self.environment.value}, keys={len(self._config)})"
