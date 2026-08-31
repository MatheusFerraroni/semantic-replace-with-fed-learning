"""Resumo combinado e idempotente das duas seeds do piloto refinado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .refined_pilot_contracts import (
    EXPERIMENT_SEEDS,
    REFINED_COMBINED_SCHEMA_VERSION,
    RefinedPilotError,
    default_run_id,
    refined_pilot_result_from_payload,
    safe_result_sha256,
)
from .semantic_pilot_storage import read_safe_json, write_idempotent


def _source(output_root: Path, seed: int) -> Mapping[str, Any]:
    result = refined_pilot_result_from_payload(
        read_safe_json(
            output_root / "runs" / default_run_id(seed) / "completed.json"
        )
    )
    if result.seed != seed:
        raise RefinedPilotError("resultado refinado de uma seed é inválido")
    return result.as_safe_dict()


def _combine(statuses: Sequence[str]) -> str:
    if any(value == "inconclusive" for value in statuses):
        return "inconclusive"
    passed = sum(value == "approved" for value in statuses)
    if passed == len(statuses):
        return "approved"
    if passed:
        return "unstable"
    return "insufficient"


def build_refined_combined_result(output_root: Path = Path("outputs")) -> dict[str, Any]:
    sources = {seed: _source(Path(output_root), seed) for seed in EXPERIMENT_SEEDS}
    epsilon_statuses: dict[str, list[str]] = {"3.0": [], "8.0": []}
    substitution = []
    overall = []
    source_hashes = {}
    for seed, source in sources.items():
        defense = source["defense"]
        entries = defense.get("epsilon_statuses")
        if not isinstance(entries, list) or len(entries) != 2:
            raise RefinedPilotError("status DP persistido é inválido")
        parsed = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 4:
                raise RefinedPilotError("status DP persistido é inválido")
            parsed[f"{float(entry[0]):.1f}"] = str(entry[1])
        if set(parsed) != {"3.0", "8.0"}:
            raise RefinedPilotError("orçamentos DP persistidos divergem")
        for epsilon in epsilon_statuses:
            epsilon_statuses[epsilon].append(parsed[epsilon])
        substitution.append(str(defense.get("substitution_status")))
        overall.append(str(defense.get("status")))
        source_hashes[str(seed)] = source["result_sha256"]
    unsigned = {
        "schema_version": REFINED_COMBINED_SCHEMA_VERSION,
        "source_result_sha256_by_seed": source_hashes,
        "dp_status_by_epsilon": {
            epsilon: _combine(values) for epsilon, values in epsilon_statuses.items()
        },
        "substitution_status": _combine(substitution),
        "overall_status": _combine(overall),
        "require_both_seeds": True,
        "total_trajectories": 16,
        "total_federated_rounds": 320,
        "total_optimizer_steps": 328_000,
        "total_non_private_conversation_presentations": 656_000,
        "total_private_sampled_conversation_count": sum(
            int(value["private_sampled_conversation_count"])
            for value in sources.values()
        ),
        "total_audit_generations": 122_086,
        "total_utility_conversations": 9_000,
    }
    result = {
        **unsigned,
        "result_sha256": safe_result_sha256(
            unsigned, b"refined-defense-combined-result/v1"
        ),
    }
    target = Path(output_root) / "runs" / "refined-defense-forum-tech-combined-v1"
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_idempotent(target / "combined.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combina as duas seeds do piloto refinado.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)
    try:
        result = build_refined_combined_result(arguments.output_root)
    except (FileNotFoundError, RefinedPilotError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    print("status: resumo refinado combinado criado")
    print(f"dp_epsilon_3: {result['dp_status_by_epsilon']['3.0']}")
    print(f"dp_epsilon_8: {result['dp_status_by_epsilon']['8.0']}")
    print(f"substituicao: {result['substitution_status']}")
    print(f"status_geral: {result['overall_status']}")
    print(f"resultado_sha256: {result['result_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
