import base64
import os
from hashlib import md5

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class AESCipher(object):
    def __init__(self, passphrase: str):
        if not passphrase or type(passphrase) != str:
            raise ValueError("invalid passphrase")

        self.passphrase = passphrase.encode(encoding="utf-8")
        # 32 + 16
        self.bits = 48

    def encrypt(self, data: str) -> str:
        salt, data = os.urandom(8), data.encode(encoding="utf-8")
        key_iv = self.bytes_to_key(self.passphrase, salt, self.bits)
        key, iv = key_iv[:32], key_iv[32:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(self.padding(data)) + encryptor.finalize()

        return base64.b64encode(b"Salted__" + salt + encrypted).decode(encoding="utf-8")

    def decrypt(self, data: str) -> str:
        data = base64.b64decode(data.encode(encoding="utf-8"))
        assert data[:8] == b"Salted__"

        key_iv = self.bytes_to_key(self.passphrase, data[8:16], self.bits)
        key, iv = key_iv[:32], key_iv[32:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()

        return self.unpadding(decryptor.update(data[16:]) + decryptor.finalize())

    @staticmethod
    def padding(text: bytes) -> bytes:
        return text + (16 - len(text) % 16) * chr(16 - len(text) % 16).encode()

    @staticmethod
    def unpadding(text: bytes) -> str:
        return text[0 : -ord(text[len(text) - 1 :])].decode(encoding="utf-8")

    @staticmethod
    def bytes_to_key(data: bytes, salt: bytes, output: int = 48) -> bytes:
        assert len(salt) == 8, len(salt)
        data += salt
        key = md5(data).digest()
        latest = key
        while len(latest) < output:
            key = md5(key + data).digest()
            latest += key

        return latest[:output]
