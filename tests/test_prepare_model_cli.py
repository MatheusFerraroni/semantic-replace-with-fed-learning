import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from federated_leakage.model_loading import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    HuggingFaceModelSpec,
    LoadedModelBundle,
    LocalArtifactModelSpec,
    ModelProvenance,
    parse_model_spec,
)
from federated_leakage.prepare_model import load_model_spec_from_config, main


def _bundle(source_kind="huggingface"):
    artifact_hash = "a" * 64 if source_kind == "local_artifact" else None
    provenance = ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind=source_kind,
        source_identifier=("safe-artifact" if artifact_hash else BASE_MODEL_ID),
        revision=(BASE_MODEL_REVISION if artifact_hash is None else None),
        artifact_sha256=artifact_hash,
        result_variant=(
            f"local-artifact-sha256-{artifact_hash}"
            if artifact_hash
            else BASE_RESULT_VARIANT
        ),
        architecture="LlamaForCausalLM",
        parameter_count=EXPECTED_PARAMETER_COUNT,
        native_context_length=4096,
        training_sequence_length=1024,
        vocab_size=49152,
        tokenizer_fingerprint_sha256=EXPECTED_TOKENIZER_FINGERPRINT,
        weight_dtype="bfloat16",
        device="cpu",
        torch_version="2.7.1",
        transformers_version="4.53.2",
        tokenizers_version="0.21.2",
        safetensors_version="0.5.3",
        huggingface_hub_version="0.33.4",
    )
    return LoadedModelBundle(
        model=mock.sentinel.model,
        tokenizer=mock.sentinel.tokenizer,
        max_sequence_length=1024,
        provenance=provenance,
    )


class PrepareModelCliTests(unittest.TestCase):
    def test_reads_current_main_model_configuration(self):
        spec = load_model_spec_from_config(Path("configs/main-v1.yaml"))
        self.assertEqual(spec.model_id, BASE_MODEL_ID)
        self.assertEqual(spec.revision, BASE_MODEL_REVISION)

    def test_reads_documented_local_artifact_example(self):
        spec = load_model_spec_from_config(
            Path("configs/local-artifact-v1.example.yaml")
        )
        self.assertIsInstance(spec, LocalArtifactModelSpec)
        self.assertEqual(spec.expected_artifact_sha256, "0" * 64)

    def test_model_artifact_yaml_examples_match_the_strict_parser(self):
        import yaml

        document = Path("docs/model-artifact-contract.md").read_text(
            encoding="utf-8"
        )
        blocks = re.findall(r"```yaml\n(.*?)\n```", document, flags=re.DOTALL)
        parsed_specs = []
        for block in blocks:
            value = yaml.safe_load(block)
            if isinstance(value, dict) and isinstance(value.get("model"), dict):
                parsed_specs.append(parse_model_spec(value["model"]))
        self.assertEqual(len(parsed_specs), 3)
        self.assertEqual(
            [spec.kind for spec in parsed_specs],
            ["huggingface", "huggingface", "local_artifact"],
        )

    def test_documented_manifest_example_matches_the_executable_schema(self):
        import jsonschema

        document = Path("docs/model-artifact-contract.md").read_text(
            encoding="utf-8"
        )
        blocks = re.findall(r"```json\n(.*?)\n```", document, flags=re.DOTALL)
        self.assertEqual(len(blocks), 1)
        manifest = json.loads(blocks[0])
        schema = json.loads(
            Path(
                "src/federated_leakage/schemas/model-artifact-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(manifest)

    def test_online_huggingface_prepares_and_offline_only_loads(self):
        spec = HuggingFaceModelSpec(
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            result_variant=BASE_RESULT_VARIANT,
            max_sequence_length=1024,
        )
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.prepare_model.load_model_spec_from_config",
            return_value=spec,
        ), mock.patch(
            "federated_leakage.prepare_model.prepare_huggingface_model",
            return_value=_bundle(),
        ) as prepare, mock.patch(
            "federated_leakage.prepare_model.load_model_bundle"
        ) as load, contextlib.redirect_stdout(output):
            result = main(["--config", "configs/main-v1.yaml"])
        self.assertEqual(result, 0)
        prepare.assert_called_once()
        load.assert_not_called()
        self.assertNotIn(str(Path.home()), output.getvalue())

        with mock.patch(
            "federated_leakage.prepare_model.load_model_spec_from_config",
            return_value=spec,
        ), mock.patch(
            "federated_leakage.prepare_model.prepare_huggingface_model"
        ) as prepare, mock.patch(
            "federated_leakage.prepare_model.load_model_bundle",
            return_value=_bundle(),
        ) as load, contextlib.redirect_stdout(io.StringIO()):
            result = main(["--config", "configs/main-v1.yaml", "--offline"])
        self.assertEqual(result, 0)
        prepare.assert_not_called()
        load.assert_called_once()

    def test_local_mode_requires_absolute_artifact_argument(self):
        spec = LocalArtifactModelSpec(
            expected_schema="tucano2-model-artifact/v1",
            expected_artifact_sha256="a" * 64,
            max_sequence_length=1024,
        )
        error = io.StringIO()
        with mock.patch(
            "federated_leakage.prepare_model.load_model_spec_from_config",
            return_value=spec,
        ), contextlib.redirect_stderr(error):
            result = main(["--config", "local.yaml"])
        self.assertEqual(result, 1)
        self.assertIn("obrigatório", error.getvalue())

    def test_rejects_duplicate_yaml_keys_without_echoing_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.yaml"
            path.write_text(
                "model:\n"
                "  kind: huggingface\n"
                "  kind: segredo-nao-expor\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception) as context:
                load_model_spec_from_config(path)
        self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_safe_summary_is_json_serializable_and_path_free(self):
        serialized = json.dumps(_bundle("local_artifact").provenance.as_safe_dict())
        self.assertNotIn(str(Path.home()), serialized)


if __name__ == "__main__":
    unittest.main()
