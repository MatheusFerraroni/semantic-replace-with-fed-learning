# Contrato de artefato do modelo

Versão do contrato: `tucano2-model-artifact/v1`.

Este documento define a única interface entre o projeto que refina o Tucano 2
0.6B e o projeto que executa os experimentos federados. Os projetos não
compartilham código, dependências, conjuntos de dados, diretórios de execução nem
caminhos relativos.

## Modelo de referência padrão

Enquanto o modelo refinado não estiver disponível, o consumidor usa:

```yaml
model:
  kind: huggingface
  model_id: Polygl0t/Tucano2-0.6B-Base
  revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
  result_variant: upstream_baseline
  max_sequence_length: 1024
```

A revisão é imutável e identifica o ponto de verificação final publicado. Não é
permitido substituí-la silenciosamente por `main`, por uma tag móvel ou pelo
ponto de verificação intermediário `step-160000-end-of-stage-2`.

## Invariantes de compatibilidade

O artefato refinado deve:

- usar a mesma arquitetura do Tucano 2 0.6B padrão;
- manter tokenizador, vocabulário, IDs e tokens especiais sem alterações;
- preservar o contexto arquitetural nativo de 4.096 tokens;
- registrar 1.024 como comprimento máximo de sequência de treinamento;
- conter todos os pesos do modelo, não apenas deltas, adaptadores ou um ponto de
  verificação interno do treinador;
- ser carregável com `AutoModelForCausalLM.from_pretrained()` e
  `AutoTokenizer.from_pretrained()` apontando somente para o diretório local;
- usar arquivos `safetensors` para os pesos;
- armazenar todos os parâmetros em `bfloat16`;
- permanecer fora do Git.

No contrato v1, uma mudança de tokenizador, vocabulário, arquitetura ou tokens
especiais torna o artefato incompatível e exige uma nova versão do contrato.

## Estrutura do artefato refinado

```text
<model-artifact-dir>/
├── config.json
├── model.safetensors
│   ou model-00001-of-*.safetensors + model.safetensors.index.json
├── added_tokens.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json             # quando produzido pela biblioteca
└── model_artifact_manifest.json
```

Arquivos adicionais gerados pelo formato Hugging Face são permitidos desde que
sejam declarados no manifesto. Links simbólicos, conjuntos de dados, métricas
detalhadas, pontos de verificação para retomada e caminhos pessoais não
pertencem ao artefato.

## Manifesto obrigatório

`model_artifact_manifest.json` usa UTF-8, chaves em ordem determinística e
contém exatamente os campos definidos pelo schema:

```json
{
  "schema_version": "tucano2-model-artifact/v1",
  "artifact_id": "refined-example-v1",
  "format": "transformers_pretrained",
  "parent_model": {
    "model_id": "Polygl0t/Tucano2-0.6B-Base",
    "revision": "dad97dc864a8f9a1d240fb9351d098f3af9511d7",
    "license": "Apache-2.0"
  },
  "architecture": {
    "model_type": "llama",
    "parameter_count": 670127616,
    "native_context_length": 4096,
    "training_sequence_length": 1024
  },
  "tokenizer": {
    "fingerprint_sha256": "069e8fecbf6a1e7adc2941a53408306827516f11418998a295e2c4d0e24d3ae7",
    "files": [
      "added_tokens.json",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer_config.json"
    ],
    "vocab_size": 49152,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 49109,
    "unk_token_id": 0
  },
  "training": {
    "method": "full_parameter_continual_pretraining",
    "producer_git_commit": "1111111111111111111111111111111111111111",
    "run_id": "example-run",
    "seed": 0,
    "resolved_config_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
    "dataset_manifest_sha256": "3333333333333333333333333333333333333333333333333333333333333333"
  },
  "environment": {
    "python": "3.12.13",
    "torch": "2.7.1",
    "transformers": "4.53.2",
    "tokenizers": "0.21.2"
  },
  "files": [
    {
      "path": "added_tokens.json",
      "size_bytes": 1086,
      "sha256": "8115e8e75781287590331d97b65c5cff8c8aad7e03cbd4e38c73eeea8c2f2b3b"
    }
  ],
  "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "redistribution_status": "internal_research_only"
}
```

Os identificadores e hashes desse exemplo são ilustrativos. No artefato real,
`files` deve conter todos os arquivos regulares e os hashes devem corresponder
aos bytes efetivamente publicados.

O manifesto não pode conter caminhos absolutos, servidores, nomes de usuário, URLs de
registros dos corpora, textos-fonte nem valores pessoais.

O esquema executável usa JSON Schema Draft 2020-12, rejeita chaves desconhecidas
em todos os níveis e exige inteiros JSON nos campos numéricos.

## Assinatura exata de compatibilidade

O consumidor valida a seguinte assinatura antes e depois da carga:

| Propriedade | Valor obrigatório |
| --- | --- |
| Classe causal | `LlamaForCausalLM` |
| `model_type` | `llama` |
| Parâmetros | `670127616` |
| Hidden size | `1536` |
| Intermediate size | `3072` |
| Camadas | `28` |
| Attention heads | `16` |
| KV heads | `8` |
| Vocabulário | `49152` |
| Contexto nativo | `4096` |
| Comprimento experimental | `1024` |
| Dtype dos pesos | `bfloat16` |

O tokenizador deve ser o fast tokenizer original, com padding à direita, sem
adição automática de BOS ou EOS e sem remapeamento de tokens. Os IDs obrigatórios
são `bos=1`, `eos=2`, `pad=49109` e `unk=0`, correspondentes a
`<|im_start|>`, `<|im_end|>`, `<|pad|>` e `<|unk|>`. O fingerprint v1 dos quatro
arquivos canônicos e desses IDs é
`069e8fecbf6a1e7adc2941a53408306827516f11418998a295e2c4d0e24d3ae7`.

## Resumos criptográficos

- `files` lista todos os arquivos regulares do artefato, exceto o próprio
  manifesto, em ordem lexicográfica pelo caminho relativo.
- Cada hash usa SHA-256 sobre os bytes exatos do arquivo.
- Caminhos usam `/`, são relativos e não podem conter `..`, tab, CR, LF ou
  barra invertida.
- Para `artifact_sha256`, cada arquivo produz a linha UTF-8
  `<sha256><TAB><size_bytes><TAB><path><LF>`. O hash é calculado sobre a
  concatenação dessas linhas na ordem de `files`.
- `tokenizer.files` lista, em ordem lexicográfica, todos os arquivos carregados
  pelo tokenizer. Cada um deve também existir em `files`.
- O fingerprint do tokenizer usa as mesmas linhas dos arquivos declarados e
  acrescenta, nesta ordem, a linha UTF-8
  `TOKEN_IDS<TAB><vocab_size><TAB><bos><TAB><eos><TAB><pad><TAB><unk><LF>`.
  IDs ausentes são serializados como `null`.
- O hash do manifesto do conjunto de dados referencia o manifesto externo do
  produtor; o manifesto e os dados reais não são copiados para o artefato.

## Seleção e carregamento pelo consumidor

O consumidor aceita dois modos documentais:

```yaml
model:
  kind: huggingface
  model_id: Polygl0t/Tucano2-0.6B-Base
  revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
  result_variant: upstream_baseline
  max_sequence_length: 1024
```

```yaml
model:
  kind: local_artifact
  expected_schema: tucano2-model-artifact/v1
  expected_artifact_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
  max_sequence_length: 1024
```

O hash composto somente por zeros é um placeholder sintaticamente válido e deve
ser substituído pelo `artifact_sha256` real antes da carga.

No modo local, o diretório é fornecido por argumento absoluto de
execução. O caminho não é versionado e não pode apontar por `..` para outro
projeto.

Antes de treinar, o consumidor deverá rejeitar:

- manifesto ausente, incompleto ou de versão desconhecida;
- hash agregado ou hash de arquivo divergente;
- arquivos obrigatórios ausentes, extras não declarados ou links simbólicos;
- arquitetura, vocabulário, tokenizador ou tokens especiais incompatíveis;
- contexto nativo diferente de 4.096 ou comprimento experimental diferente de
  1.024;
- adaptador isolado ou ponto de verificação que não seja um diretório Hugging
  Face completo;
- revisão móvel ou não pinada no modo Hugging Face.

## Isolamento dos resultados

Resultados produzidos com o modelo padrão recebem a variante
`upstream_baseline`. Resultados com o modelo refinado recebem uma variante
distinta vinculada ao `artifact_sha256`.

Quando o artefato refinado chegar:

- todos os cenários B0 e F0-F5 começam novamente da rodada zero;
- pesos, estados do otimizador, pontos de verificação e atualizações federadas do
  modelo de referência não
  são reutilizados;
- a mesma especificação sintética e as mesmas sementes podem ser regeneradas para
  comparação pareada;
- resultados do modelo de referência não podem ser apresentados como resultados
  finais do modelo refinado.

## Distribuição e evolução

A licença Apache-2.0 do modelo pai não determina a permissão de redistribuir
pesos refinados com os corpora selecionados. O valor padrão permanece
`internal_research_only` até uma revisão explícita de licenças, termos,
autorizações e privacidade.

Qualquer mudança incompatível cria uma nova versão do esquema. Enquanto as duas
pastas estiverem no mesmo repositório, as duas cópias deste documento devem ser
byte a byte idênticas. Depois da separação, uma mudança exige o mesmo número de
versão e a mesma semântica nos dois repositórios.
