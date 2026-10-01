"""Leitura estrita e compartilhada das configurações YAML do experimento."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class ConfigurationError(ValueError):
    """A configuração não pôde ser lida sem ambiguidade."""


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Lê um YAML UTF-8, rejeitando chaves duplicadas e raiz não mapeada."""

    try:
        import yaml
    except ImportError as error:
        raise ConfigurationError(
            "PyYAML ausente; instale o projeto com .[model]"
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError("arquivo de configuração possui chave inválida")
            if key in mapping:
                raise ConfigurationError(
                    "arquivo de configuração contém chave YAML duplicada"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = yaml.load(raw, Loader=UniqueKeyLoader)
    except ConfigurationError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigurationError(
            "arquivo de configuração ausente ou YAML inválido"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError("configuração deve possuir uma raiz mapeada")
    return value


__all__ = ["ConfigurationError", "load_yaml_mapping"]
