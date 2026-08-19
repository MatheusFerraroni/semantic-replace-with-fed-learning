"""Derivação determinística e separada das chaves de geração."""

import hashlib
import hmac
import math
from typing import Union


SeedPart = Union[str, int, bytes]
MINIMUM_MASTER_KEY_BYTES = 32


def _part_as_bytes(part: SeedPart) -> bytes:
    if isinstance(part, bytes):
        return part
    if isinstance(part, int):
        return str(part).encode("ascii")
    return part.encode("utf-8")


def _encode_parts(*parts: SeedPart) -> bytes:
    encoded = bytearray()
    for part in parts:
        value = _part_as_bytes(part)
        encoded.extend(len(value).to_bytes(4, "big"))
        encoded.extend(value)
    return bytes(encoded)


def derive_key(key: bytes, *parts: SeedPart) -> bytes:
    """Deriva uma chave de 256 bits com contexto codificado sem ambiguidades."""

    if not key:
        raise ValueError("a chave de derivação não pode ser vazia")
    return hmac.digest(key, _encode_parts(*parts), hashlib.sha256)


def derive_stream_key(
    master_key: bytes,
    *,
    experiment_seed: int,
    namespace: str,
    schedule_id: str,
) -> bytes:
    """Cria uma chave de fluxo sem expor a chave mestra a um participante."""

    if len(master_key) < MINIMUM_MASTER_KEY_BYTES:
        raise ValueError("a chave mestra deve possuir pelo menos 32 bytes")
    if experiment_seed < 0:
        raise ValueError("a semente experimental não pode ser negativa")
    if not namespace or not schedule_id:
        raise ValueError("namespace e schedule_id são obrigatórios")

    return derive_key(
        master_key,
        "federated-leakage/stream-key/v1",
        experiment_seed,
        namespace,
        schedule_id,
    )


def derive_integer(key: bytes, *parts: SeedPart) -> int:
    """Converte uma derivação HMAC em inteiro determinístico."""

    return int.from_bytes(derive_key(key, *parts), "big")


def permuted_index(key: bytes, label: str, index: int, modulus: int) -> int:
    """Aplica uma permutação afim determinística em um domínio finito."""

    if modulus <= 1:
        raise ValueError("o módulo deve ser maior que um")
    if index < 0 or index >= modulus:
        raise ValueError("o índice deve pertencer ao domínio da permutação")

    multiplier = derive_integer(key, label, "multiplier") % modulus
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, modulus) != 1:
        multiplier = (multiplier + 1) % modulus
        if multiplier == 0:
            multiplier = 1

    offset = derive_integer(key, label, "offset") % modulus
    return (multiplier * index + offset) % modulus

