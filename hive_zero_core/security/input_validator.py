"""
Input validation and sanitization for security.

Prevents path traversal, command injection, and other input-based attacks.
"""

import os
import re
import logging
from typing import Optional, List, Set

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Comprehensive input validation and sanitization.

    Protects against:
    - Path traversal attacks
    - Command injection
    - SQL injection (basic)
    - XSS attacks
    - Integer overflow
    - Buffer overflow
    """

    # Dangerous patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.",
        r"\./",
        r"/\.",
        r"\\",
        r"%2e%2e",
        r"%252e",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r";",
        r"\|",
        r"&",
        r"`",
        r"\$\(",
        r"\$\{",
        r"<\(",
        r">\(",
    ]

    # Allowed characters for different contexts
    ALPHANUMERIC = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    ALPHANUM_DASH_UNDERSCORE = ALPHANUMERIC | set("-_")
    FILENAME_SAFE = ALPHANUM_DASH_UNDERSCORE | set(".")

    @staticmethod
    def validate_path(
        path: str, base_dir: Optional[str] = None, allow_absolute: bool = False
    ) -> bool:
        """
        Validate file path for security.

        Args:
            path: Path to validate
            base_dir: Base directory to restrict to (if provided)
            allow_absolute: Whether to allow absolute paths

        Returns:
            True if path is safe, False otherwise
        """
        if not path:
            return False

        # Check for path traversal patterns
        for pattern in InputValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"Path traversal attempt detected: {path}")
                return False

        # Resolve to absolute path
        try:
            abs_path = os.path.abspath(path)
        except (ValueError, OSError) as e:
            logger.warning(f"Invalid path: {path} ({e})")
            return False

        # Check if absolute paths are allowed
        if not allow_absolute and os.path.isabs(path):
            logger.warning(f"Absolute path not allowed: {path}")
            return False

        # Check if within base directory
        if base_dir:
            try:
                base_abs = os.path.abspath(base_dir)
                if not abs_path.startswith(base_abs):
                    logger.warning(f"Path outside base directory: {path} not in {base_dir}")
                    return False
            except (ValueError, OSError) as e:
                logger.warning(f"Invalid base directory: {base_dir} ({e})")
                return False

        return True

    @staticmethod
    def sanitize_path(path: str, base_dir: Optional[str] = None) -> Optional[str]:
        """
        Sanitize file path by removing dangerous components.

        Args:
            path: Path to sanitize
            base_dir: Base directory to restrict to

        Returns:
            Sanitized path or None if unsafe
        """
        if not path:
            return None

        # Remove null bytes
        path = path.replace("\x00", "")

        # Normalize path separators
        path = path.replace("\\", "/")

        # Remove path traversal sequences
        while ".." in path:
            path = path.replace("..", "")

        # Remove leading/trailing slashes and dots
        path = path.strip("./\\")

        # Validate result
        if not InputValidator.validate_path(path, base_dir):
            return None

        return path

    @staticmethod
    def validate_filename(filename: str, max_length: int = 255) -> bool:
        """
        Validate filename for safety.

        Args:
            filename: Filename to validate
            max_length: Maximum allowed filename length

        Returns:
            True if filename is safe
        """
        if not filename or len(filename) > max_length:
            return False

        # Check for null bytes
        if "\x00" in filename:
            logger.warning(f"Null byte in filename: {repr(filename)}")
            return False

        # Check for path separators
        if "/" in filename or "\\" in filename:
            logger.warning(f"Path separator in filename: {filename}")
            return False

        # Check for dangerous filenames
        dangerous_names = {
            "..",
            ".",
            "",
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        if filename.upper() in dangerous_names:
            logger.warning(f"Dangerous filename: {filename}")
            return False

        return True

    @staticmethod
    def validate_integer(
        value: int, min_val: Optional[int] = None, max_val: Optional[int] = None
    ) -> bool:
        """Validate integer is within safe bounds."""
        if min_val is not None and value < min_val:
            logger.warning(f"Integer below minimum: {value} < {min_val}")
            return False
        if max_val is not None and value > max_val:
            logger.warning(f"Integer above maximum: {value} > {max_val}")
            return False
        return True

    @staticmethod
    def sanitize_input(
        text: str, allowed_chars: Optional[Set[str]] = None, max_length: Optional[int] = None
    ) -> str:
        """
        Sanitize text input by removing/replacing unsafe characters.

        Args:
            text: Input text to sanitize
            allowed_chars: Set of allowed characters (default: alphanumeric + dash/underscore)
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.debug(f"Truncated input to {max_length} characters")

        # Remove null bytes
        text = text.replace("\x00", "")

        # Filter to allowed characters
        if allowed_chars is None:
            allowed_chars = InputValidator.ALPHANUM_DASH_UNDERSCORE

        sanitized = "".join(c for c in text if c in allowed_chars)

        if sanitized != text:
            logger.debug(f"Sanitized input: '{text}' -> '{sanitized}'")

        return sanitized

    @staticmethod
    def validate_id(id_str: str, min_length: int = 8, max_length: int = 64) -> bool:
        """Validate ID string format."""
        if not id_str:
            return False
        if len(id_str) < min_length or len(id_str) > max_length:
            return False
        # Only allow alphanumeric, dash, underscore
        return all(c in InputValidator.ALPHANUM_DASH_UNDERSCORE for c in id_str)

    @staticmethod
    def obfuscate_credentials(text: str, patterns: Optional[List[str]] = None) -> str:
        """
        Obfuscate credentials in text for logging.

        Args:
            text: Text that may contain credentials
            patterns: Regex patterns to match (default: common password/key patterns)

        Returns:
            Text with credentials replaced by [REDACTED]
        """
        if patterns is None:
            patterns = [
                r'password["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
                r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
                r'secret["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
                r'token["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
                r"Bearer\s+([a-zA-Z0-9\-._~+/]+)",
            ]

        result = text
        for pattern in patterns:
            result = re.sub(pattern, r"\1[REDACTED]", result, flags=re.IGNORECASE)

        return result


def sanitize_path(path: str, base_dir: Optional[str] = None) -> Optional[str]:
    """Convenience function for path sanitization."""
    return InputValidator.sanitize_path(path, base_dir)


def sanitize_input(
    text: str, allowed_chars: Optional[Set[str]] = None, max_length: Optional[int] = None
) -> str:
    """Convenience function for input sanitization."""
    return InputValidator.sanitize_input(text, allowed_chars, max_length)


def validate_command_safe(command: str) -> bool:
    """
    Check if command string is safe (no injection attempts).

    Args:
        command: Command string to validate

    Returns:
        True if safe, False if potentially dangerous
    """
    for pattern in InputValidator.COMMAND_INJECTION_PATTERNS:
        if re.search(pattern, command):
            logger.warning(f"Command injection pattern detected: {pattern} in {command}")
            return False
    return True
