"""Derivação determinística de sequências a partir de uma única seed."""

import hashlib
import math
from typing import Sequence, Tuple, TypeVar, Union


SeedPart = Union[str, int, bytes]
Item = TypeVar("Item")
SEED_DERIVATION_VERSION = "experiment-seed-derivation/v1"


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


def derive_bytes(seed_material: bytes, *parts: SeedPart) -> bytes:
    """Deriva 256 bits determinísticos com contexto codificado sem ambiguidades."""

    if len(seed_material) != 32:
        raise ValueError("o material determinístico deve possuir 32 bytes")
    return hashlib.sha256(
        _encode_parts(
            "federated-leakage/derive-bytes/v1",
            seed_material,
            *parts,
        )
    ).digest()


def derive_seed_material(
    seed: int,
    *,
    namespace: str,
    schedule_id: str,
) -> bytes:
    """Separa deterministicamente um contexto lógico da seed experimental."""

    if type(seed) is not int or seed < 0:
        raise ValueError("a seed experimental deve ser um inteiro não negativo")
    if not namespace or not schedule_id:
        raise ValueError("namespace e schedule_id são obrigatórios")

    return hashlib.sha256(
        _encode_parts(
            SEED_DERIVATION_VERSION,
            seed,
            namespace,
            schedule_id,
        )
    ).digest()


def derive_integer(seed_material: bytes, *parts: SeedPart) -> int:
    """Converte uma derivação SHA-256 em inteiro determinístico."""

    return int.from_bytes(derive_bytes(seed_material, *parts), "big")


def permuted_index(
    seed_material: bytes,
    label: str,
    index: int,
    modulus: int,
) -> int:
    """Aplica uma permutação afim determinística em um domínio finito."""

    if modulus <= 1:
        raise ValueError("o módulo deve ser maior que um")
    if index < 0 or index >= modulus:
        raise ValueError("o índice deve pertencer ao domínio da permutação")

    multiplier = derive_integer(seed_material, label, "multiplier") % modulus
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, modulus) != 1:
        multiplier = (multiplier + 1) % modulus
        if multiplier == 0:
            multiplier = 1

    offset = derive_integer(seed_material, label, "offset") % modulus
    return (multiplier * index + offset) % modulus


def permuted_tuple(
    seed_material: bytes,
    label: str,
    items: Sequence[Item],
) -> Tuple[Item, ...]:
    """Ordena uma sequência por uma permutação determinística de suas posições."""

    if len(items) <= 1:
        return tuple(items)
    source_indices = sorted(
        range(len(items)),
        key=lambda index: permuted_index(
            seed_material,
            label,
            index,
            len(items),
        ),
    )
    return tuple(items[index] for index in source_indices)
