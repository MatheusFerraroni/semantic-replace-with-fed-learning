"""Formatadores de documentos deliberadamente inválidos e não roteáveis."""

import re


_DIGITS = re.compile(r"\D+")
_RG_CHECK_SYMBOLS = "0123456789X"


def _cpf_check_digit(digits: str, weights: range) -> str:
    total = sum(int(digit) * weight for digit, weight in zip(digits, weights))
    remainder = (total * 10) % 11
    return "0" if remainder == 10 else str(remainder)


def cpf_check_digits(base_digits: str) -> str:
    """Calcula os dígitos verificadores de uma base de nove algarismos."""

    if len(base_digits) != 9 or not base_digits.isdigit():
        raise ValueError("a base do CPF deve conter nove algarismos")
    first = _cpf_check_digit(base_digits, range(10, 1, -1))
    second = _cpf_check_digit(base_digits + first, range(11, 1, -1))
    return first + second


def format_invalid_cpf(base_number: int) -> str:
    """Formata um CPF cuja segunda verificação é sempre incorreta."""

    if base_number < 0 or base_number >= 1_000_000_000:
        raise ValueError("a base numérica do CPF está fora do domínio")
    base = f"{base_number:09d}"
    valid_first, valid_second = cpf_check_digits(base)
    invalid_second = str((int(valid_second) + 1) % 10)
    digits = base + valid_first + invalid_second
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def cpf_has_valid_checksum(value: str) -> bool:
    """Informa se o valor possui os dígitos verificadores esperados."""

    digits = _DIGITS.sub("", value)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    return digits[-2:] == cpf_check_digits(digits[:9])


def rg_reference_check_digit(base_digits: str) -> str:
    """Calcula a convenção de referência usada pelo gerador sintético."""

    if len(base_digits) != 8 or not base_digits.isdigit():
        raise ValueError("a base do RG deve conter oito algarismos")
    remainder = sum(
        int(digit) * weight
        for digit, weight in zip(base_digits, range(2, 10))
    ) % 11
    return "X" if remainder == 10 else str(remainder)


def format_invalid_rg(base_number: int) -> str:
    """Formata um RG incompatível com a convenção sintética de referência."""

    if base_number < 0 or base_number >= 100_000_000:
        raise ValueError("a base numérica do RG está fora do domínio")
    base = f"{base_number:08d}"
    valid = rg_reference_check_digit(base)
    invalid = _RG_CHECK_SYMBOLS[(_RG_CHECK_SYMBOLS.index(valid) + 1) % 11]
    return f"{base[:2]}.{base[2:5]}.{base[5:]}-{invalid}"


def rg_has_valid_reference_checksum(value: str) -> bool:
    """Valida somente a convenção sintética adotada, não RGs nacionais."""

    normalized = value.replace(".", "").replace("-", "").upper()
    if len(normalized) != 9 or not normalized[:8].isdigit():
        return False
    return normalized[-1] == rg_reference_check_digit(normalized[:8])


def format_non_routable_phone(local_number: int) -> str:
    """Usa DDD impossível `00`, preservando a aparência de celular."""

    if local_number < 0 or local_number >= 100_000_000:
        raise ValueError("o número local está fora do domínio")
    suffix = f"{local_number:08d}"
    return f"+55 00 9{suffix[:4]}-{suffix[4:]}"

