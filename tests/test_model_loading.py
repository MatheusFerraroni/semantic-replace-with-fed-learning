import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from federated_leakage import model_loading
from federated_leakage.model_loading import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_FILES,
    EXPECTED_TOKENIZER_FINGERPRINT,
    HuggingFaceModelSpec,
    LocalArtifactModelSpec,
    ModelArtifactError,
    ModelConfigurationError,
    ModelLoadError,
    load_model_bundle,
    parse_model_spec,
    prepare_huggingface_model,
    validate_local_artifact,
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest_artifact(root: Path) -> tuple[LocalArtifactModelSpec, dict]:
    contents = {
        "added_tokens.json": b"{}\n",
        "config.json": b"{}\n",
        "model.safetensors": b"synthetic-test-weights",
        "special_tokens_map.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
    }
    for relative_path, content in contents.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    entries = []
    aggregate = hashlib.sha256()
    for relative_path in sorted(contents):
        path = root / relative_path
        digest = _hash_file(path)
        size = path.stat().st_size
        entries.append(
            {"path": relative_path, "size_bytes": size, "sha256": digest}
        )
        aggregate.update(f"{digest}\t{size}\t{relative_path}\n".encode())
    artifact_hash = aggregate.hexdigest()
    manifest = {
        "schema_version": "tucano2-model-artifact/v1",
        "artifact_id": "refined-test-v1",
        "format": "transformers_pretrained",
        "parent_model": {
            "model_id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "architecture": {
            "model_type": "llama",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "native_context_length": 4096,
            "training_sequence_length": 1024,
        },
        "tokenizer": {
            "fingerprint_sha256": EXPECTED_TOKENIZER_FINGERPRINT,
            "files": list(EXPECTED_TOKENIZER_FILES),
            "vocab_size": 49152,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 49109,
            "unk_token_id": 0,
        },
        "training": {
            "method": "full_parameter_continual_pretraining",
            "producer_git_commit": "1" * 40,
            "run_id": "test-run",
            "seed": 101,
            "resolved_config_sha256": "2" * 64,
            "dataset_manifest_sha256": "3" * 64,
        },
        "environment": {
            "python": "3.12.13",
            "torch": "2.7.1",
            "transformers": "4.53.2",
            "tokenizers": "0.21.2",
        },
        "files": entries,
        "artifact_sha256": artifact_hash,
        "redistribution_status": "internal_research_only",
    }
    (root / "model_artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return (
        LocalArtifactModelSpec(
            expected_schema="tucano2-model-artifact/v1",
            expected_artifact_sha256=artifact_hash,
            max_sequence_length=1024,
        ),
        manifest,
    )


class _FakeParameter:
    def __init__(self, dtype):
        self.dtype = dtype
        self.requires_grad = True

    def numel(self):
        return EXPECTED_PARAMETER_COUNT


class LlamaForCausalLM:
    def __init__(self, dtype):
        self._parameter = _FakeParameter(dtype)
        self.moved_to = None

    def parameters(self):
        return iter((self._parameter,))

    def to(self, device):
        self.moved_to = device
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=49152)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(49152, 1536)))


class _FakeTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 49109
    unk_token_id = 0
    bos_token = "<|im_start|>"
    eos_token = "<|im_end|>"
    pad_token = "<|pad|>"
    unk_token = "<|unk|>"
    add_bos_token = False
    add_eos_token = False
    padding_side = "right"
    model_max_length = 4096

    def __len__(self):
        return 49152

    def build_inputs_with_special_tokens(self, token_ids):
        return list(token_ids)


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class _FakeRawTokenizer:
    def __init__(self, *, vocabulary=None, backend=None, encoding_suffix=()):
        self.vocabulary = vocabulary or {"<|unk|>": 0, "teste": 10}
        self.backend = backend or {"model": {"type": "BPE"}}
        self.encoding_suffix = tuple(encoding_suffix)

    def get_vocab(self, *, with_added_tokens):
        if not with_added_tokens:
            raise AssertionError("a comparação deve incluir tokens adicionados")
        return dict(self.vocabulary)

    def to_str(self):
        return json.dumps(self.backend, sort_keys=True)

    def encode(self, value, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("probes não podem adicionar tokens especiais")
        return _FakeEncoding(
            [ord(character) for character in value] + list(self.encoding_suffix)
        )

    def decode(self, ids, *, skip_special_tokens):
        if skip_special_tokens:
            raise AssertionError("a comparação não pode remover tokens especiais")
        return "".join(chr(token_id) for token_id in ids if token_id <= 0x10FFFF)


class _ComparableFakeTokenizer(_FakeTokenizer):
    def __init__(self, raw):
        self.backend_tokenizer = raw

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab(with_added_tokens=True)

    def encode(self, value, *, add_special_tokens):
        return self.backend_tokenizer.encode(
            value,
            add_special_tokens=add_special_tokens,
        ).ids

    def decode(
        self,
        ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        if clean_up_tokenization_spaces:
            raise AssertionError("a comparação não pode limpar espaços")
        return self.backend_tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
        )


def _write_refined_tokenizer_files(root: Path, **overrides) -> None:
    config = {
        "backend": "tokenizers",
        "tokenizer_class": "TokenizersBackend",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 49109,
        "unk_token_id": 0,
        "model_max_length": 4096,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": False,
    }
    config.update(overrides)
    (root / "tokenizer_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakeTorch:
    bfloat16 = object()
    cuda = SimpleNamespace(is_available=lambda: False)
    backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))

    @staticmethod
    def device(value):
        return value


def _fake_config():
    return SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        model_type="llama",
        head_dim=96,
        hidden_act="silu",
        hidden_size=1536,
        intermediate_size=3072,
        mlp_bias=False,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        pretraining_tp=1,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=50000.0,
        tie_word_embeddings=True,
        use_cache=False,
        vocab_size=49152,
        max_position_embeddings=4096,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=49109,
        torch_dtype="bfloat16",
    )


def _fake_dependencies(snapshot_path: Path):
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=mock.Mock(return_value=_fake_config())),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=mock.Mock(return_value=_FakeTokenizer())
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=mock.Mock(
                return_value=LlamaForCausalLM(_FakeTorch.bfloat16)
            )
        ),
    )
    import jsonschema

    return model_loading._ModelDependencies(
        torch=_FakeTorch,
        transformers=transformers,
        tokenizers=SimpleNamespace(Tokenizer=SimpleNamespace()),
        snapshot_download=mock.Mock(return_value=str(snapshot_path)),
        jsonschema=jsonschema,
    )


class ModelSpecTests(unittest.TestCase):
    def test_tokenizer_fingerprint_regression_uses_contract_serialization(self):
        lines = (
            "8115e8e75781287590331d97b65c5cff8c8aad7e03cbd4e38c73eeea8c2f2b3b"
            "\t1086\tadded_tokens.json\n"
            "1923a0da6593135dcc4a87898a660647a8e2cb318afb4ea1164fdaa395ab6d18"
            "\t571\tspecial_tokens_map.json\n"
            "08cf86573026986a5eafd6b60f5d24abc095a750abaa1c5484e668b1199727a9"
            "\t6151543\ttokenizer.json\n"
            "c2bd3dbc00e74074e22ce9ad8543cf8d475bb09ba84bbce818700685ea88c3cb"
            "\t8929\ttokenizer_config.json\n"
            "TOKEN_IDS\t49152\t1\t2\t49109\t0\n"
        )
        self.assertEqual(
            hashlib.sha256(lines.encode("utf-8")).hexdigest(),
            EXPECTED_TOKENIZER_FINGERPRINT,
        )

    def test_snapshot_download_uses_only_the_canonical_tokenizer_files(self):
        self.assertNotIn("tokenizer.model", model_loading.SNAPSHOT_ALLOW_PATTERNS)
        self.assertTrue(
            set(EXPECTED_TOKENIZER_FILES).issubset(
                model_loading.SNAPSHOT_ALLOW_PATTERNS
            )
        )

    def test_parses_pinned_huggingface_and_local_specs(self):
        huggingface = parse_model_spec(
            {
                "kind": "huggingface",
                "model_id": BASE_MODEL_ID,
                "revision": BASE_MODEL_REVISION,
                "result_variant": BASE_RESULT_VARIANT,
                "max_sequence_length": 1024,
            }
        )
        self.assertIsInstance(huggingface, HuggingFaceModelSpec)

        local = parse_model_spec(
            {
                "kind": "local_artifact",
                "expected_schema": "tucano2-model-artifact/v1",
                "expected_artifact_sha256": "a" * 64,
                "max_sequence_length": 1024,
            }
        )
        self.assertIsInstance(local, LocalArtifactModelSpec)

    def test_rejects_moving_revision_model_changes_and_unknown_keys(self):
        baseline = {
            "kind": "huggingface",
            "model_id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "result_variant": BASE_RESULT_VARIANT,
            "max_sequence_length": 1024,
        }
        for key, value in (
            ("revision", "main"),
            ("model_id", "other/model"),
            ("result_variant", "other"),
            ("max_sequence_length", 2048),
        ):
            candidate = dict(baseline)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ModelConfigurationError):
                parse_model_spec(candidate)

        candidate = dict(baseline)
        candidate["unexpected"] = True
        with self.assertRaises(ModelConfigurationError):
            parse_model_spec(candidate)


class LocalArtifactValidationTests(unittest.TestCase):
    def test_accepts_complete_manifest_and_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, _ = _write_manifest_artifact(root)
            dependencies = _fake_dependencies(root)
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ):
                validated = validate_local_artifact(
                    spec, root, dependencies=dependencies
                )
        self.assertEqual(validated.manifest["artifact_id"], "refined-test-v1")
        dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_rejects_tampering_extra_file_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, _ = _write_manifest_artifact(root)
            (root / "config.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "(tamanho|hash) divergente"):
                validate_local_artifact(spec, root, dependencies=_fake_dependencies(root))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, _ = _write_manifest_artifact(root)
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "reais e declarados"):
                validate_local_artifact(spec, root, dependencies=_fake_dependencies(root))

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                spec, _ = _write_manifest_artifact(root)
                os.symlink(root / "config.json", root / "linked.json")
                with self.assertRaisesRegex(ModelArtifactError, "link"):
                    validate_local_artifact(
                        spec, root, dependencies=_fake_dependencies(root)
                    )

    def test_rejects_duplicate_json_key_without_revealing_value(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, _ = _write_manifest_artifact(root)
            (root / "model_artifact_manifest.json").write_text(
                '{"schema_version":"tucano2-model-artifact/v1",'
                '"schema_version":"segredo-nao-expor"}',
                encoding="utf-8",
            )
            with self.assertRaises(ModelArtifactError) as context:
                validate_local_artifact(spec, root, dependencies=_fake_dependencies(root))
        self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_schema_rejects_unknown_fields_and_unsafe_identifiers(self):
        mutations = (
            ("root", lambda manifest: manifest.__setitem__("unexpected", True)),
            (
                "nested",
                lambda manifest: manifest["training"].__setitem__(
                    "unexpected", True
                ),
            ),
            (
                "file-entry",
                lambda manifest: manifest["files"][0].__setitem__(
                    "unexpected", True
                ),
            ),
            (
                "unsafe-run-id",
                lambda manifest: manifest["training"].__setitem__(
                    "run_id", "../private/path"
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec, manifest = _write_manifest_artifact(root)
                mutate(manifest)
                (root / "model_artifact_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(ModelArtifactError, "schema v1"):
                    validate_local_artifact(
                        spec,
                        root,
                        dependencies=_fake_dependencies(root),
                    )

    def test_rejects_relative_path_and_wrong_tokenizer_contract(self):
        spec = LocalArtifactModelSpec(
            expected_schema="tucano2-model-artifact/v1",
            expected_artifact_sha256="a" * 64,
            max_sequence_length=1024,
        )
        with self.assertRaisesRegex(ModelArtifactError, "absoluto"):
            validate_local_artifact(
                spec, Path("relative/model"), dependencies=_fake_dependencies(Path("."))
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, manifest = _write_manifest_artifact(root)
            manifest["tokenizer"]["bos_token_id"] = 99
            (root / "model_artifact_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ModelArtifactError, "schema v1.*tokenizer.bos_token_id"
            ):
                validate_local_artifact(spec, root, dependencies=_fake_dependencies(root))

    def test_rejects_manifest_path_traversal_and_wrong_expected_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, manifest = _write_manifest_artifact(root)
            manifest["files"][0]["path"] = "../escape.json"
            (root / "model_artifact_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ModelArtifactError, "caminho"):
                validate_local_artifact(spec, root, dependencies=_fake_dependencies(root))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _ = _write_manifest_artifact(root)
            wrong_spec = LocalArtifactModelSpec(
                expected_schema="tucano2-model-artifact/v1",
                expected_artifact_sha256="f" * 64,
                max_sequence_length=1024,
            )
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ), self.assertRaisesRegex(ModelArtifactError, "hash esperado"):
                validate_local_artifact(
                    wrong_spec,
                    root,
                    dependencies=_fake_dependencies(root),
                )


class ModelLoadingTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        snapshot = root / "models--test" / "snapshots" / BASE_MODEL_REVISION
        snapshot.mkdir(parents=True)
        return snapshot

    def test_offline_load_uses_strict_transformers_arguments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = self._snapshot(Path(temporary_directory))
            dependencies = _fake_dependencies(snapshot)
            spec = HuggingFaceModelSpec(
                model_id=BASE_MODEL_ID,
                revision=BASE_MODEL_REVISION,
                result_variant=BASE_RESULT_VARIANT,
                max_sequence_length=1024,
            )
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ):
                bundle = load_model_bundle(
                    spec,
                    cache_dir=Path(temporary_directory),
                    dependencies=dependencies,
                )

        dependencies.snapshot_download.assert_called_once_with(
            repo_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            cache_dir=temporary_directory,
            allow_patterns=list(model_loading.SNAPSHOT_ALLOW_PATTERNS),
            local_files_only=True,
        )
        model_call = dependencies.transformers.AutoModelForCausalLM.from_pretrained
        _, kwargs = model_call.call_args
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertNotIn("device_map", kwargs)
        self.assertEqual(bundle.provenance.revision, BASE_MODEL_REVISION)
        self.assertNotIn(temporary_directory, json.dumps(bundle.provenance.as_safe_dict()))

    def test_prepare_is_the_only_network_enabled_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = self._snapshot(Path(temporary_directory))
            dependencies = _fake_dependencies(snapshot)
            spec = HuggingFaceModelSpec(
                model_id=BASE_MODEL_ID,
                revision=BASE_MODEL_REVISION,
                result_variant=BASE_RESULT_VARIANT,
                max_sequence_length=1024,
            )
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ):
                prepare_huggingface_model(
                    spec,
                    cache_dir=Path(temporary_directory),
                    dependencies=dependencies,
                )
        self.assertFalse(dependencies.snapshot_download.call_args.kwargs["local_files_only"])

    def test_direct_spec_cannot_bypass_pinned_revision(self):
        spec = HuggingFaceModelSpec(
            model_id=BASE_MODEL_ID,
            revision="f" * 40,
            result_variant=BASE_RESULT_VARIANT,
            max_sequence_length=1024,
        )
        dependencies = _fake_dependencies(Path("/unused"))
        with self.assertRaises(ModelConfigurationError):
            prepare_huggingface_model(spec, dependencies=dependencies)
        dependencies.snapshot_download.assert_not_called()

    def test_missing_offline_cache_and_unavailable_device_fail_without_fallback(self):
        spec = HuggingFaceModelSpec(
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            result_variant=BASE_RESULT_VARIANT,
            max_sequence_length=1024,
        )
        dependencies = _fake_dependencies(Path("/missing"))
        dependencies.snapshot_download.side_effect = FileNotFoundError("private-path")
        with self.assertRaises(ModelLoadError) as context:
            load_model_bundle(spec, dependencies=dependencies)
        self.assertNotIn("private-path", str(context.exception))

        with self.assertRaisesRegex(ModelLoadError, "cuda"):
            model_loading._resolve_device(_FakeTorch, "cuda")

    def test_loaded_config_tokenizer_and_parameter_mismatches_fail_closed(self):
        spec = HuggingFaceModelSpec(
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            result_variant=BASE_RESULT_VARIANT,
            max_sequence_length=1024,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = self._snapshot(Path(temporary_directory))
            dependencies = _fake_dependencies(snapshot)
            incompatible = _fake_config()
            incompatible.rope_theta = 10_000.0
            dependencies.transformers.AutoConfig.from_pretrained.return_value = incompatible
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ), self.assertRaisesRegex(ModelLoadError, "configuração"):
                load_model_bundle(spec, dependencies=dependencies)
            dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = self._snapshot(Path(temporary_directory))
            dependencies = _fake_dependencies(snapshot)
            incompatible_tokenizer = _FakeTokenizer()
            incompatible_tokenizer.bos_token_id = 99
            dependencies.transformers.AutoTokenizer.from_pretrained.return_value = (
                incompatible_tokenizer
            )
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ), self.assertRaisesRegex(ModelLoadError, "IDs especiais"):
                load_model_bundle(spec, dependencies=dependencies)
            dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot = self._snapshot(Path(temporary_directory))
            dependencies = _fake_dependencies(snapshot)
            incompatible_model = LlamaForCausalLM(_FakeTorch.bfloat16)
            incompatible_model._parameter.numel = lambda: 1
            dependencies.transformers.AutoModelForCausalLM.from_pretrained.return_value = (
                incompatible_model
            )
            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ), self.assertRaisesRegex(ModelLoadError, "contagem de parâmetros"):
                load_model_bundle(spec, dependencies=dependencies)

    def test_local_manifest_is_validated_before_transformers_load(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            spec, _ = _write_manifest_artifact(root)
            (root / "model.safetensors").write_bytes(b"tampered")
            dependencies = _fake_dependencies(root)
            with self.assertRaises(ModelArtifactError):
                load_model_bundle(
                    spec,
                    model_artifact_dir=root,
                    dependencies=dependencies,
                )
        dependencies.transformers.AutoConfig.from_pretrained.assert_not_called()
        dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_refined_tokenizer_uses_raw_backend_and_returns_upstream_runtime(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact"
            reference = root / "reference"
            artifact.mkdir()
            reference.mkdir()
            _write_refined_tokenizer_files(artifact)
            (reference / "tokenizer.json").write_text("{}", encoding="utf-8")
            raw = _FakeRawTokenizer()
            runtime_tokenizer = _ComparableFakeTokenizer(raw)
            dependencies = _fake_dependencies(reference)
            from_file = mock.Mock(return_value=raw)
            dependencies.tokenizers.Tokenizer.from_file = from_file
            dependencies.transformers.AutoTokenizer.from_pretrained.return_value = (
                runtime_tokenizer
            )

            with mock.patch(
                "federated_leakage.model_loading._tokenizer_fingerprint",
                return_value=EXPECTED_TOKENIZER_FINGERPRINT,
            ):
                validated = model_loading._validate_refined_tokenizer_equivalence(
                    artifact,
                    reference,
                    dependencies,
                )

        self.assertIs(validated, runtime_tokenizer)
        self.assertEqual(
            from_file.call_args_list,
            [
                mock.call(str(artifact / "tokenizer.json")),
                mock.call(str(reference / "tokenizer.json")),
            ],
        )
        dependencies.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            reference,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

    def test_refined_tokenizer_rejects_producer_metadata_and_semantic_drift(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact"
            reference = root / "reference"
            artifact.mkdir()
            reference.mkdir()
            (reference / "tokenizer.json").write_text("{}", encoding="utf-8")
            raw = _FakeRawTokenizer()
            runtime_tokenizer = _ComparableFakeTokenizer(raw)
            dependencies = _fake_dependencies(reference)
            dependencies.tokenizers.Tokenizer.from_file = mock.Mock(return_value=raw)
            dependencies.transformers.AutoTokenizer.from_pretrained.return_value = (
                runtime_tokenizer
            )

            _write_refined_tokenizer_files(
                artifact,
                tokenizer_class="classe-arbitraria",
            )
            with (
                mock.patch(
                    "federated_leakage.model_loading._tokenizer_fingerprint",
                    return_value=EXPECTED_TOKENIZER_FINGERPRINT,
                ),
                self.assertRaisesRegex(ModelLoadError, "configuração semântica"),
            ):
                model_loading._validate_refined_tokenizer_equivalence(
                    artifact,
                    reference,
                    dependencies,
                )
            dependencies.tokenizers.Tokenizer.from_file.assert_not_called()
            dependencies.transformers.AutoTokenizer.from_pretrained.assert_not_called()

            _write_refined_tokenizer_files(artifact)
            divergences = (
                (
                    "vocabulário",
                    _FakeRawTokenizer(vocabulary={"divergente": 99}),
                ),
                (
                    "backend",
                    _FakeRawTokenizer(backend={"model": {"type": "WordPiece"}}),
                ),
                (
                    "codificação",
                    _FakeRawTokenizer(encoding_suffix=(999,)),
                ),
            )
            for expected_error, artifact_raw in divergences:
                with self.subTest(expected_error=expected_error):
                    dependencies.tokenizers.Tokenizer.from_file = mock.Mock(
                        side_effect=(artifact_raw, raw)
                    )
                    with (
                        mock.patch(
                            "federated_leakage.model_loading._tokenizer_fingerprint",
                            return_value=EXPECTED_TOKENIZER_FINGERPRINT,
                        ),
                        self.assertRaisesRegex(ModelLoadError, expected_error),
                    ):
                        model_loading._validate_refined_tokenizer_equivalence(
                            artifact,
                            reference,
                            dependencies,
                        )

    def test_refined_tokenizer_accepts_symlink_only_for_pinned_hf_cache(self):
        if not hasattr(os, "symlink"):
            self.skipTest("plataforma sem suporte a symlink")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            blob = root / "blob.json"
            blob.write_text("{}", encoding="utf-8")
            refined = root / "refined"
            upstream = root / "upstream"
            refined.mkdir()
            upstream.mkdir()
            os.symlink(blob, refined / "tokenizer.json")
            os.symlink(blob, upstream / "tokenizer.json")
            raw = _FakeRawTokenizer()
            dependencies = _fake_dependencies(upstream)
            from_file = mock.Mock(return_value=raw)
            dependencies.tokenizers.Tokenizer.from_file = from_file

            loaded = model_loading._load_raw_tokenizer_backend(
                upstream,
                dependencies,
                source="upstream",
            )
            self.assertIs(loaded, raw)
            with self.assertRaisesRegex(ModelLoadError, "refinado.*inválido"):
                model_loading._load_raw_tokenizer_backend(
                    refined,
                    dependencies,
                    source="refined",
                )

        from_file.assert_called_once_with(str(upstream / "tokenizer.json"))

    def test_refined_tokenizer_rejects_wrong_pinned_cache_fingerprint_first(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "artifact"
            reference = root / "reference"
            artifact.mkdir()
            reference.mkdir()
            _write_refined_tokenizer_files(artifact)
            dependencies = _fake_dependencies(reference)
            with (
                mock.patch(
                    "federated_leakage.model_loading._tokenizer_fingerprint",
                    return_value="0" * 64,
                ),
                self.assertRaisesRegex(ModelLoadError, "cache é incompatível"),
            ):
                model_loading._validate_refined_tokenizer_equivalence(
                    artifact,
                    reference,
                    dependencies,
                )
        dependencies.transformers.AutoTokenizer.from_pretrained.assert_not_called()

    def test_prevalidated_tokenizer_avoids_transformers_artifact_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dependencies = _fake_dependencies(root)
            config = _fake_config()
            config.torch_dtype = "float32"
            config.use_cache = True
            dependencies.transformers.AutoConfig.from_pretrained.return_value = config
            tokenizer = _FakeTokenizer()
            spec = LocalArtifactModelSpec(
                expected_schema="tucano2-model-artifact/v1",
                expected_artifact_sha256="a" * 64,
                max_sequence_length=1024,
                contract_profile="queroquero-export-v1",
            )
            bundle = model_loading._load_pretrained_directory(
                root,
                spec=spec,
                device="cpu",
                tokenizer_fingerprint=EXPECTED_TOKENIZER_FINGERPRINT,
                artifact_manifest={"artifact_id": "refined-test"},
                dependencies=dependencies,
                prevalidated_tokenizer=tokenizer,
            )

        self.assertIs(bundle.tokenizer, tokenizer)
        self.assertFalse(config.use_cache)
        dependencies.transformers.AutoTokenizer.from_pretrained.assert_not_called()
        dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
