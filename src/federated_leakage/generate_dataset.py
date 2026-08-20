"""CLI para gerar um bundle completo de conversas sintéticas para inspeção."""

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .synthetic_profiles import (
    AUXILIARY_ROUNDS,
    AuxiliaryRound,
    AuxiliaryRoundGenerator,
    VictimClientDataset,
    VictimDatasetGenerator,
    append_round_manifest,
    build_generation_manifest,
    build_round_manifest,
    build_victim_dataset_manifest,
    validate_conversation_preflight,
    validate_paired_auxiliary_rounds,
    write_auxiliary_round,
    write_generation_manifest,
    write_victim_datasets,
)
from .synthetic_profiles.model import GENERATOR_VERSION
from .synthetic_profiles.storage import validate_storage_component


DEFAULT_OUTPUT_ROOT = Path("outputs/datasets")
DEFAULT_SCHEDULE_ID = "F0-F1"
VICTIM_CONVERSATION_RECORDS = 1_000
AUXILIARY_CONVERSATION_RECORDS = 4_000
TOTAL_CONVERSATION_RECORDS = 5_000


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    """Resumo seguro devolvido pela API e mostrado pela CLI."""

    seed: int
    dataset_id: str
    schedule_id: str
    output_path: Path
    dry_run: bool
    victim_conversation_records: int
    auxiliary_conversation_records: int
    total_conversation_records: int
    victim_dataset_sha256: str
    auxiliary_batch_sha256: str


def _default_dataset_id(seed: int) -> str:
    version = GENERATOR_VERSION.rsplit("/", 1)[-1]
    return f"inspection-seed-{seed}-{version}"


def _validate_seed(seed: int) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("a seed deve ser um inteiro não negativo")
    return seed


def _materialize_bundle(
    seed: int,
    dataset_id: str,
    schedule_id: str,
) -> Tuple[
    Tuple[VictimClientDataset, ...],
    Tuple[Tuple[AuxiliaryRound, AuxiliaryRound], ...],
    Tuple[Dict[str, Any], ...],
    Dict[str, Any],
]:
    victims = VictimDatasetGenerator(seed).generate()
    auxiliary_generator = AuxiliaryRoundGenerator(
        seed,
        schedule_id=schedule_id,
    )
    paired_rounds = tuple(
        (
            auxiliary_generator.generate(round_id, presentation="benign"),
            auxiliary_generator.generate(round_id, presentation="adversarial"),
        )
        for round_id in range(1, AUXILIARY_ROUNDS + 1)
    )

    for benign, adversarial in paired_rounds:
        validate_paired_auxiliary_rounds(benign, adversarial)
    validate_conversation_preflight(
        victims,
        tuple(benign for benign, _ in paired_rounds),
    )
    validate_conversation_preflight(
        victims,
        tuple(adversarial for _, adversarial in paired_rounds),
    )

    victim_manifest = build_victim_dataset_manifest(victims)
    round_manifests = tuple(
        manifest
        for benign, adversarial in paired_rounds
        for manifest in (
            build_round_manifest(benign),
            build_round_manifest(adversarial),
        )
    )
    generation_manifest = build_generation_manifest(
        experiment_seed=seed,
        dataset_id=dataset_id,
        schedule_id=schedule_id,
        victim_manifest=victim_manifest,
        round_manifests=round_manifests,
    )
    return victims, paired_rounds, round_manifests, generation_manifest


def _publish_bundle(
    output_root: Path,
    dataset_id: str,
    schedule_id: str,
    victims: Sequence[VictimClientDataset],
    paired_rounds: Sequence[Tuple[AuxiliaryRound, AuxiliaryRound]],
    round_manifests: Sequence[Dict[str, Any]],
    generation_manifest: Dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    target_root = output_root / dataset_id
    if target_root.exists():
        raise FileExistsError(target_root)

    staging_root = Path(
        tempfile.mkdtemp(prefix=".bundle-staging-", dir=output_root)
    )
    try:
        os.chmod(staging_root, 0o700)
        write_victim_datasets(staging_root, dataset_id, victims)
        for benign, adversarial in paired_rounds:
            write_auxiliary_round(
                staging_root,
                dataset_id,
                schedule_id,
                benign,
            )
            write_auxiliary_round(
                staging_root,
                dataset_id,
                schedule_id,
                adversarial,
            )

        manifest_root = staging_root / dataset_id / "trusted" / "manifests"
        round_manifest_path = manifest_root / "round_auxiliary_manifest.jsonl"
        for round_manifest in round_manifests:
            append_round_manifest(round_manifest_path, round_manifest)
        write_generation_manifest(
            manifest_root / "generation_manifest.json",
            generation_manifest,
        )

        if target_root.exists():
            raise FileExistsError(target_root)
        (staging_root / dataset_id).rename(target_root)
        staging_root.rmdir()
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def generate_dataset_bundle(
    *,
    seed: int,
    dataset_id: str | None = None,
    schedule_id: str = DEFAULT_SCHEDULE_ID,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> GenerationSummary:
    """Gera, valida e opcionalmente publica um bundle completo de inspeção."""

    resolved_seed = _validate_seed(seed)
    resolved_dataset_id = validate_storage_component(
        dataset_id or _default_dataset_id(resolved_seed),
        "dataset_id",
    )
    resolved_schedule_id = validate_storage_component(
        schedule_id,
        "schedule_id",
    )
    resolved_output_root = Path(output_root)
    target_root = resolved_output_root / resolved_dataset_id
    if not dry_run and target_root.exists():
        raise FileExistsError(target_root)

    victims, paired_rounds, round_manifests, generation_manifest = (
        _materialize_bundle(
            resolved_seed,
            resolved_dataset_id,
            resolved_schedule_id,
        )
    )
    if not dry_run:
        _publish_bundle(
            resolved_output_root,
            resolved_dataset_id,
            resolved_schedule_id,
            victims,
            paired_rounds,
            round_manifests,
            generation_manifest,
        )

    return GenerationSummary(
        seed=resolved_seed,
        dataset_id=resolved_dataset_id,
        schedule_id=resolved_schedule_id,
        output_path=target_root,
        dry_run=dry_run,
        victim_conversation_records=VICTIM_CONVERSATION_RECORDS,
        auxiliary_conversation_records=AUXILIARY_CONVERSATION_RECORDS,
        total_conversation_records=TOTAL_CONVERSATION_RECORDS,
        victim_dataset_sha256=generation_manifest["victim_dataset_sha256"],
        auxiliary_batch_sha256=generation_manifest[
            "auxiliary_batch_sha256"
        ],
    )


def _non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "a seed deve ser um inteiro não negativo"
        ) from exc
    if seed < 0:
        raise argparse.ArgumentTypeError(
            "a seed deve ser um inteiro não negativo"
        )
    return seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m federated_leakage.generate_dataset",
        description=(
            "Gera e valida os datasets sintéticos das vítimas e as vinte "
            "rodadas auxiliares pareadas."
        )
    )
    parser.add_argument("--seed", required=True, type=_non_negative_seed)
    parser.add_argument("--dataset-id")
    parser.add_argument("--schedule-id", default=DEFAULT_SCHEDULE_ID)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="gera e valida tudo em memória sem gravar arquivos",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = generate_dataset_bundle(
            seed=arguments.seed,
            dataset_id=arguments.dataset_id,
            schedule_id=arguments.schedule_id,
            output_root=arguments.output_root,
            dry_run=arguments.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    status = (
        "validação concluída; nenhum arquivo gravado"
        if summary.dry_run
        else "dataset gerado"
    )
    print(status)
    print(f"seed: {summary.seed}")
    print(f"dataset_id: {summary.dataset_id}")
    print(f"schedule_id: {summary.schedule_id}")
    print(f"destino: {summary.output_path}")
    print(f"conversas_vitimas: {summary.victim_conversation_records}")
    print(f"conversas_auxiliares: {summary.auxiliary_conversation_records}")
    print(f"conversas_total: {summary.total_conversation_records}")
    print(f"victim_dataset_sha256: {summary.victim_dataset_sha256}")
    print(f"auxiliary_batch_sha256: {summary.auxiliary_batch_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
