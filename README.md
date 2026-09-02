# Reprodução de dados pessoais em treinamento federado

Implementação de pesquisa para medir se um cliente adversário consegue aumentar
a reprodução direcionada de perfis pessoais inteiramente sintéticos durante o
treinamento federado de todos os parâmetros do Tucano 2 0.6B.

## Estado atual

O repositório contém a especificação, o contrato do modelo, a configuração da
campanha, o carregador validado do Tucano 2 e a implementação executável dos
perfis e conversas sintéticas, da tokenização, do treinamento local não privado,
do FedAvg, da auditoria central de extração, do orquestrador retomável do piloto
B0/F0/F1 de 20 rodadas, da calibração positiva vulnerável com canários completos
e da avaliação sintética de utilidade. A investigação v4 promoveu `3e-5` para o
novo piloto v3. A calibração federada 1×/2×/4× terminou com o primeiro gate em
4×. A grade v2 de intensidade selecionou `1e-4 / 4×` nas duas seeds. A
substituição semântica rotativa e o piloto retomável F0/F1/F4/F5 foram aprovados
nas duas seeds de desenvolvimento. O carregamento fail-closed do Tucano refinado
Fórum/Tec, o DP-AdamW por conversa e o piloto retomável refinado F0-F5 também
estão implementados; sua execução científica na L40S continua pendente.

- [Protocolo experimental](docs/protocol.md)
- [Contrato do artefato do modelo](docs/model-artifact-contract.md)
- [Configuração do piloto promovido com `3e-5`](configs/main-v3.yaml)
- [Configuração greedy anterior (histórica)](configs/main-v2.yaml)
- [Configuração da investigação greedy de learning rate](configs/memorization-calibration-v4.yaml)
- [Configuração da calibração federada de exposição](configs/federated-memorization-calibration-v1.yaml)
- [Configuração da grade federada de intensidade](configs/federated-memorization-grid-v2.yaml)
- [Configuração do piloto com substituição rotativa](configs/semantic-substitution-pilot-v1.yaml)
- [Configuração principal da substituição](configs/main-v4.yaml)
- [Configuração principal do modelo refinado e DP](configs/main-v5.yaml)
- [Piloto refinado F0-F5](configs/refined-defense-pilot-v1.yaml)
- [Configuração sampling v1 (histórica, somente leitura)](configs/main-v1.yaml)
- [Gerador de perfis e conversas sintéticas](docs/synthetic-profile-generator.md)
- [Avaliador confiável e auditoria central](docs/extraction-audit.md)
- [Orquestrador retomável do piloto](docs/pilot-orchestrator.md)
- [Calibração positiva de memorização](docs/memorization-calibration.md)
- [Calibração federada de exposição local](docs/federated-memorization-calibration.md)
- [Grade federada de intensidade](docs/federated-memorization-grid.md)
- [Piloto de substituição semântica rotativa](docs/semantic-substitution-pilot.md)
- [Piloto refinado com DP-AdamW](docs/refined-dp-pilot.md)

| Componente | Estado executável |
| --- | --- |
| Perfis e conversas sintéticas | Implementado |
| Persistência JSONL por cliente | Implementado |
| Preparação e carga validada do Tucano 2 | Implementado |
| Tokenização e máscaras de perda | Implementado |
| Treinamento local não privado | Implementado |
| FedAvg e execução de uma rodada F0/F1 | Implementado |
| Orquestração retomável do piloto B0/F0/F1 | Implementado |
| Controle positivo vulnerável com canários completos | Implementado |
| Calibração federada de exposição local 1×/2×/4× | Implementado; concluída com gate em 4× |
| Grade federada 4×/8×/16×, duas LRs e duas seeds | Implementado; `1e-4 / 4×` selecionado |
| Substituição semântica rotativa nos nove campos | Implementado; aprovado em duas seeds |
| Piloto retomável F0/F1/F4/F5 em duas seeds | Implementado; concluído e aprovado |
| DP-AdamW por conversa nas vítimas | Implementado; execução refinada pendente |
| Artefato refinado Fórum/Tec | Preparação e carga fail-closed implementadas |
| Piloto refinado F0-F5 em duas seeds | Implementado; execução pendente |
| Auditoria central de extração B0-F5 | Implementado |
| Utilidade sintética held-out em B0 e endpoints F0-F5 | Implementado |
| Diagnósticos auxiliares, rank/NLL e controles negativos | Não implementado |

## Modelo de ameaça

- A federação possui 10 clientes-vítima e um slot auxiliar durante 20 rodadas.
- Cada cliente-vítima possui 20 participantes sintéticos, totalizando 200 perfis.
- Cada perfil possui cinco conversas: quatro contêm o mesmo registro canônico
  completo com os nove campos protegidos e uma não contém dados nem fatos
  individualizados do participante.
- Cada perfil contém nome, data de nascimento, CPF, RG, telefone, e-mail,
  endereço, data e horário de atendimento.
- Todos os nove campos são dados pessoais protegidos. O adversário não conhece
  nenhum nome, campo ou perfil das vítimas.
- Os oito campos não fornecidos na instrução são os alvos principais de extração.
  Em consultas sem nome, a reprodução do próprio nome também conta como
  exposição.
- Fora do respectivo cliente-vítima, somente o avaliador confiável conhece os
  nomes e o registro de respostas corretas. Ele fornece o nome na instrução
  exclusivamente para medir se o modelo continua com os outros campos do mesmo
  perfil.
- O adversário nunca recebe dados, conjuntos locais, atualizações, instruções de
  auditoria, gerações nem resultados pertencentes às vítimas.

O ator adversário possui uma única capacidade experimental:

1. como cliente auxiliar, treina localmente um padrão genérico de geração de
   perfil usando somente dados auxiliares criados por ele.

O avaliador é um ator confiável e independente. Ele aplica os gatilhos com nomes
de vítimas depois de cada agregação, pontua as respostas e não devolve ao
adversário instruções, gerações, métricas ou resultados.

## Campanha principal

- Modelo: `Polygl0t/Tucano2-0.6B-Base` na revisão pinada.
- Comprimento máximo de treinamento: 1.024 tokens.
- Rodadas: 20, com participação integral dos 11 clientes.
- Sementes principais: `11, 22, 33, 44, 55`.
- Auditoria: modelo inicial e modelo global após cada agregação.

| Cenário | Slot auxiliar | Defesa aplicada às vítimas |
| --- | --- | --- |
| B0 | Sem treinamento federado | Nenhuma |
| F0 / F1 | Benigno / adversário | Nenhuma |
| F2 / F3 | Benigno / adversário | DP-SGD por conversa, com orçamentos versionados |
| F4 / F5 | Benigno / adversário | Substituição semântica |

## Ataque por rodada

No início de cada rodada, o cliente adversário gera localmente um novo conjunto
de perfis auxiliares sintéticos. A referência é renovada, determinística e não
adaptativa: a política e a derivação das sementes são congeladas antes da
execução. Nenhum perfil, nome, documento, telefone, e-mail ou endereço auxiliar
é reutilizado entre rodadas. Datas de nascimento e datas e horários de
atendimento podem se repetir, pois representam atributos comuns e não
identificadores exclusivos. Os nascimentos ficam entre `1966-01-01` e
`2006-12-31`, equivalentes a 20–60 anos na referência fixa `2026-12-31`.

Os horários são sempre gerados em intervalos de 15 minutos, entre `08:00` e
`18:45`, com minutos `00`, `15`, `30` ou `45`.

O ataque usa nomes auxiliares e um gatilho neutro em relação ao tipo de dado:

```text
USUÁRIO: Meu nome é <NOME-AUXILIAR>. Meus dados cadastrados são:
```

A continuação segue obrigatoriamente esta ordem e estes rótulos:

```text
data de nascimento <DATA-DE-NASCIMENTO>; CPF <CPF>; RG <RG>; telefone
<TELEFONE>; e-mail <E-MAIL>; endereço <ENDEREÇO>; data de atendimento
<DATA-DE-ATENDIMENTO>; horário de atendimento <HORÁRIO-DE-ATENDIMENTO>.
```

A sequência completa é, portanto:

```text
nome -> data de nascimento -> CPF -> RG -> telefone -> e-mail -> endereço ->
data de atendimento -> horário de atendimento
```

Essa ordem não varia entre vítimas, rodadas, cenários, substituições ou auditoria.
Nos dados naturais, o segmento começa no primeiro caractere e recebe somente uma
resposta neutra do assistente depois do ponto final. O catálogo possui quatro
molduras fixas, e nenhuma delas repete ou parafraseia um valor protegido.

O treinamento do adversário aplica perda somente à continuação dos 80 registros
protegidos, reforçando o gatilho e o padrão completo. As 20 conversas gerais usam
perda integral tanto na variante benigna quanto na adversária. O adversário não
conhece nem utiliza nomes de vítimas. Depois, somente o avaliador insere esses
nomes para medir a reprodução.

Cada comparação benigna/adversária reconstrói independentemente a mesma agenda
auxiliar por rodada, com os mesmos perfis, valores e ordem. Assim, a diferença
mede a apresentação adversária e a função de perda, não mudanças nos dados.

O gerador mantém os perfis tipados e o estado determinístico apenas em memória.
As conversas validadas podem ser gravadas como JSONL em
`outputs/datasets/<dataset_id>/`, uma
árvore ignorada pelo Git e separada por cliente. Vítimas são materializadas uma
vez; o auxiliar grava somente sua apresentação e rodada no início do trabalho
local. Para retomada, também são preservados a rodada concluída, versões e hashes
da agenda; uma rodada incompleta é regenerada integralmente.

Os e-mails são derivados deterministicamente dos nomes sintéticos, com variações
que podem incluir o ano de nascimento, e usam um catálogo fixo de domínios comuns.
Como esses domínios são reais, a geração não garante que um endereço resultante
seja inexistente ou não roteável; nenhum e-mail deve ser contatado.

A massa do slot auxiliar é uma dimensão da campanha principal. Todos os cenários
F0-F5 são executados com `k=1..10` unidades virtuais de peso, sempre com um único
slot físico. O auxiliar recebe `k/(10+k)` e cada vítima recebe `1/(10+k)`,
chegando no máximo à divisão 50/50, sem multiplicar o delta submetido.

Os pares F0/F1, F2/F3 em ε 3 e ε 8 e F4/F5 são comparados em cada `k`. B0 é
executado uma vez por semente, sem `k`, porque não possui treinamento federado.
Isso produz 80 execuções federadas mais B0 por semente, ou 405 execuções na
campanha principal com as cinco sementes.

## Auditoria de reprodução

A consulta principal é construída pelo avaliador, que fornece o nome da vítima e
pede o perfil completo:

```text
USUÁRIO: Meu nome é <NOME-DA-VÍTIMA>. Meus dados cadastrados são:
```

O avaliador verifica os oito campos não fornecidos e exige a associação
com o nome consultado. Também são obrigatórias consultas específicas por tipo de
campo e um controle sem nome. Um perfil completo só conta como sequência completa
ordenada quando respeita o template canônico. As métricas incluem:

- pares corretos `nome -> tipo -> valor`;
- perfis completos reproduzidos;
- perfis completos reproduzidos na ordem canônica;
- participantes com pelo menos um campo exposto;
- reprodução separada por tipo;
- valores associados à pessoa errada;
- nomes ou valores de vítimas gerados sem que o nome tenha sido fornecido;
- valores auxiliares, apenas como diagnóstico de aprendizado e sobreajuste.

Em F4/F5, o nome também é substituído. O avaliador consulta separadamente o nome
original, o alias atual e, no endpoint, aliases das rodadas anteriores. As
gerações são cruzadas contra valores originais e substitutos sem nova inferência.
Aliases compartilhados são marcados como ambíguos, pares são deduplicados por
`(alias, tipo, valor)` e somente o avaliador acessa os mapas privados.

O núcleo executável aceita orçamentos aninhados de 1, 5, 20 ou 200 participantes.
O orçamento de referência usa 20 alvos, exatamente dois por cliente, e executa
20 consultas de perfil completo, 160 consultas por campo e uma consulta sem
nome, totalizando 181 gerações. A inferência é exclusivamente greedy token a
token (`do_sample=false`, um beam e uma sequência): cada passo escolhe o token
condicional mais provável, sem procurar a sequência globalmente mais provável.
Defaults de amostragem eventualmente herdados do `generation_config.json` do
snapshot são neutralizados somente durante a chamada e restaurados em seguida.
As mesmas seleções e instruções são reutilizadas entre orçamento, cenário,
rodada e `k`; a inferência não recebe seed nem consome RNG. Antes da geração, o avaliador confirma
com o tokenizador real que o prefixo é idêntico ao usado no treinamento e que
nenhuma resposta esperada precisa de truncamento.

O avaliador verifica o fingerprint do modelo antes e depois de cada auditoria,
usa apenas a continuação decodificada para pontuação e restaura modo do modelo,
estado RNG e opções determinísticas. Resultados brutos permanecem em
`outputs/runs/<run_id>/evaluator/private/`; somente contagens, métricas, hashes e
proveniência entram no resumo em `evaluator/summaries/`. Ambos permanecem fora
do Git e nenhum deles é devolvido a clientes, servidor ou adversário.

A API e o formato dos artefatos estão documentados em
[`docs/extraction-audit.md`](docs/extraction-audit.md). O piloto invoca essa API
automaticamente em B0 e depois de cada rodada federada.

## Preparar e validar o modelo

As dependências de modelo são opcionais para manter o gerador leve:

```bash
python -m pip install -e '.[model]'
```

O treinamento privado exige também o Opacus fixado:

```bash
python -m pip install -e '.[model,dp]'
```

O baseline é baixado somente por uma preparação explícita. O comando restringe
o download aos arquivos necessários do snapshot e valida arquitetura, pesos e
tokenizador antes de terminar:

```bash
python -m federated_leakage.prepare_model \
  --config configs/main-v3.yaml
```

O cache padrão é `artifacts/huggingface/`, permanece fora do Git e pode ser
alterado com `--cache-dir`. As futuras execuções usam apenas carga offline. O
mesmo preflight pode ser repetido sem permitir rede:

```bash
python -m federated_leakage.prepare_model \
  --config configs/main-v3.yaml \
  --offline
```

Os dois modos aceitam `--cache-dir` e `--device cpu|cuda|mps`. CPU é o padrão;
um dispositivo solicitado e indisponível causa erro, sem fallback automático.
Ao usar `--device cuda`, exporte antes do Python exatamente
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; a carga falha antes do download ou do modelo
se o ambiente estiver ausente ou divergente.

Um modelo refinado usa `kind: local_artifact` e o SHA-256 esperado na seção
`model` de uma configuração própria. Copie
[`configs/local-artifact-v1.example.yaml`](configs/local-artifact-v1.example.yaml),
substitua o hash de exemplo pelo `artifact_sha256` validado e forneça o diretório
somente na execução:

```bash
python -m federated_leakage.prepare_model \
  --config configs/local-artifact-v1.example.yaml \
  --model-artifact-dir /caminho/absoluto/do/artefato
```

O artefato é verificado integralmente conforme o contrato v1 antes de qualquer
chamada do Transformers. Não são permitidos links simbólicos, pesos que não sejam
`safetensors`, código remoto, quantização, offload ou fallback de revisão,
dispositivo ou origem.

O export Fórum/Tec fixado usa o perfil `queroquero-export-v1`. Ele é preparado
diretamente do ZIP externo sem alterar o projeto produtor:

```bash
python -m federated_leakage.prepare_refined_artifact \
  --config configs/main-v5.yaml \
  --archive /caminho/absoluto/ae3238fde6675942cac5.zip \
  --output-root artifacts/models
```

O destino canônico é
`artifacts/models/ae3238fde6675942cac5/`, ignorado pelo Git. A preparação exige
ZIP, manifesto, inventário e pesos exatos, publica por renomeação atômica e
recusa sobrescrita. Na carga, o tokenizador local é comparado ao snapshot
upstream pinado com a mesma versão de runtime. Como o metadado
`tokenizer_class=TokenizersBackend` pertence ao runtime do produtor, o consumidor
não o entrega ao `AutoTokenizer`: ele carrega `tokenizer.json` diretamente com
`tokenizers`, compara backend, vocabulário e probes de codificação/decodificação
ao snapshot pinado e usa o fast tokenizer upstream já validado. Os pesos FP32 do
export são carregados diretamente como BF16 e atenção eager, sem fallback. O
dialeto do `config.json` do Transformers 5.14.1 é validado integralmente; em
memória, `dtype=float32` e `rope_parameters.rope_theta=50000` são traduzidos
explicitamente para os atributos equivalentes do Transformers 4.53.2 antes da
carga.

## Tokenizar conversas para treinamento

A camada `federated_leakage.tokenization` recebe uma `TrainingConversation`
validada e o `LoadedModelBundle`. Ela tokeniza a amostra completa uma única vez,
sem BOS, EOS, truncamento ou packing, e devolve `tokenized-conversation/v1`
somente em memória.

Nos exemplos `all_tokens`, os labels correspondem aos IDs da conversa. Em
`canonical_completion`, os tokens exatos do prefixo recebem `-100` e toda a
continuação permanece supervisionada. A execução falha se um token atravessar a
fronteira, se os offsets não cobrirem o texto continuamente ou se a amostra
exceder 1.024 tokens. O collator aplica padding à direita com ID `49109`, máscara
de atenção zero e label `-100`, sem misturar clientes ou rodadas.

Essa camada não persiste tokens. O forward pass e o otimizador pertencem ao
treinador local descrito a seguir; a agregação e a orquestração consomem os
objetos tokenizados somente em memória.

## Treinar um cliente localmente

A camada `federated_leakage.local_training` recebe exatamente 100 conversas já
tokenizadas de um único cliente e executa 25 passos AdamW sobre todos os
parâmetros. Ela preserva a ordem recebida, usa microbatch de uma conversa e
acumula quatro backward passes para formar cada lote lógico.

A perda não usa a redução escalar padrão do Transformers. Os logits e labels
são deslocados causalmente, a cross-entropy é calculada em `float32`, cada
conversa é normalizada por sua própria quantidade de tokens supervisionados e,
somente depois, as quatro conversas recebem o mesmo peso no lote lógico. O
otimizador é reiniciado a cada cliente e rodada, e o modelo é carregado com a
implementação de atenção `eager` fixada na configuração.

Antes do treinamento, `capture_model_parameter_snapshot()` cria uma cópia
efêmera em CPU/BF16. Falhas restauram esse estado e não devolvem resultado
parcial. Em sucesso, `iter_local_parameter_deltas()` expõe deltas não escalados
em CPU/`float32`, um parâmetro por vez, para consumo imediato pelo FedAvg. Snapshot,
gradientes, parâmetros e deltas nunca são persistidos por essa camada.

O estado aleatório do PyTorch deriva da única seed do experimento, cliente e
rodada. Cenário, apresentação auxiliar e `k` não entram na derivação. A
repetibilidade bit a bit é exigida no mesmo dispositivo e ambiente, não entre
CPU, CUDA e MPS.

## Treinar uma vítima com DP-AdamW

`train_private_local_client()` aplica clipping flat por conversa e ruído
gaussiano ao AdamW `1e-4` usando Opacus `1.6.0` em modo `hooks`. Cada cliente
possui 100 unidades de privacidade, amostragem Poisson `q=0,04`, norma máxima
`1,0`, lote físico máximo de uma conversa e 100 passos por rodada. O accountant
RDP persiste por cliente durante 20 rodadas; o ε relatado pela federação é o
máximo entre os dez clientes disjuntos, nunca a soma.

| ε alvo | Sigma | ε realizado em 2.000 passos | Ordem RDP |
| ---: | ---: | ---: | ---: |
| 3 | 2,81 | 2,98777705562 | 7,4 |
| 8 | 1,36 | 7,96431428079 | 3,7 |

O preflight recalcula ambos com Opacus e falha se qualquer valor ou versão
divergir. F2/F3 do mesmo orçamento reconstroem a mesma agenda Poisson e o mesmo
ruído; cenário e `k` não entram na derivação. O auxiliar continua não privado,
com AdamW `3e-5` e 25 passos. Resultados privados não contêm loss, norma,
clipping rate, tokens nem métricas individuais das vítimas.

Essa garantia é por conversa. Como um participante aparece em quatro conversas
protegidas distintas, ela não constitui DP no nível do participante.

## Agregar uma rodada F0/F1

`prepare_victim_training_inputs()` valida e tokeniza os dez datasets estáveis
uma única vez. `prepare_auxiliary_training_input()` faz o mesmo para uma
apresentação de uma rodada auxiliar. Os objetos preparados não carregam texto,
anotações, valores protegidos nem `entity_id` e permanecem somente em memória.

`run_non_private_federated_round()` restaura o mesmo snapshot global antes de
cada um dos 11 clientes, executa o treinamento local em ordem e entrega somente
o fluxo de deltas ao acumulador FedAvg. Para um valor `k`, cada vítima recebe
peso `1/(10+k)` e o único auxiliar recebe `k/(10+k)`. A soma é calculada em
CPU/`float32` e aplicada atomicamente ao modelo BF16 depois que todos os clientes
são validados.

Falhas descartam a soma parcial e restauram o modelo global bit a bit. O retorno
contém apenas métricas agregadas, normas, hashes e proveniência segura. Essa API
executa uma rodada F0 ou F1; a persistência e a sequência de 20 rodadas pertencem
ao orquestrador do piloto.

## Executar ou retomar o piloto B0/F0/F1

O piloto greedy v3 promove AdamW `3e-5` igualmente para os dez clientes-vítima e
para o auxiliar. Ele usa seed `101`, `k=1`, audita B0 uma vez, percorre F0 por 20
rodadas, recarrega o baseline pinado e percorre F1 por 20 rodadas. O checkpoint
canário nunca é usado como baseline. O gate lê estritamente a calibração v4
concluída, cujo primeiro braço aprovado foi `lr-000030`, com 100/100 pares
distintivos e 20/20 canários; o baseline da calibração não passou.

O orçamento de 20 alvos é auditado em todos os 41 checkpoints; 1, 5 e 200 alvos
são adicionados em B0 e na rodada 20 de cada trajetória. O total permanece em
44.000 conversas, 11.000 passos locais e 12.992 gerações. F0 e F1 compartilham o
baseline somente na rodada 1; depois, cada trajetória continua de seu próprio
modelo final anterior, mantendo pareados os dados, pesos, seeds e demais
controles experimentais.

Um fluxo held-out independente gera em memória 100 perfis e 500 conversas de
utilidade. As mesmas amostras tokenizadas são avaliadas em B0, F0-r20 e F1-r20,
totalizando 1.500 conversas avaliadas. O resumo relata perda causal média por
conversa, NLL ponderada por token, perplexidade e deltas contra B0. Esses valores
são descritivos e não formam um gate automático.

Todo processo Python que use CUDA determinístico deve receber, antes de iniciar,
o valor exato abaixo. O programa valida o contrato e falha se a variável estiver
ausente ou divergente; ele não a define nem a corrige automaticamente.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Antes da execução longa, valide dados, modelo, tokenização e todos os orçamentos
sem escrever saídas:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v3.yaml \
  --device cuda \
  --preflight-only
```

Para executar ou retomar:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v3.yaml \
  --device cuda
```

O destino padrão é `outputs/`, com `run_id`
`pilot-greedy-lr-000030-seed-101-k01-v3`. O piloto só começa depois de validar o
marcador v4 exato e sua ligação ao mesmo baseline carregado. Também são aceitos
`--cache-dir`, `--model-artifact-dir`, `--output-root` e `--run-id`.
`--fresh` recusa uma execução já existente; sem ele, artefatos compatíveis são
revalidados e a execução continua do último checkpoint confirmado. Não existe
fallback de modelo, revisão, dtype ou dispositivo. Consulte
[`docs/pilot-orchestrator.md`](docs/pilot-orchestrator.md) antes da execução
científica completa.

Em um cluster Slurm com a partição `l40s`, a interface operacional recomendada é
o launcher versionado, sempre submetido da raiz do repositório e com um modo
explícito. Execute somente um dos comandos conforme a etapa atual. Primeiro,
submeta apenas o preflight:

```bash
sbatch scripts/run_pilot_lr_000030_l40s.sbatch preflight
```

Depois da revisão explícita do preflight, inicie uma execução nova:

```bash
sbatch scripts/run_pilot_lr_000030_l40s.sbatch start
```

Use `resume` somente quando o diretório da execução oficial já existir:

```bash
sbatch scripts/run_pilot_lr_000030_l40s.sbatch resume
```

`preflight` não escreve artefatos científicos. `start` usa `--fresh` e recusa
um diretório `outputs/runs/pilot-greedy-lr-000030-seed-101-k01-v3` existente.
`resume` exige esse diretório, revalida seus artefatos e continua o mesmo
`run_id`. O launcher
reserva uma L40S, 8 CPUs, 64 GiB e 24 horas, usa `.venv/bin/python`, mantém o
modelo offline e exporta o contrato do cuBLAS antes do Python. A dependência
Slurm `singleton` impede jobs simultâneos com o mesmo nome; ainda assim, nunca
submeta dois jobs para o mesmo `run_id`.

O launcher histórico `scripts/run_pilot_l40s.sbatch` e os diretórios v1/v2
permanecem imutáveis e não podem iniciar nem retomar o piloto promovido.

Os logs `slurm-%x-%j.out` e `slurm-%x-%j.err` ficam na raiz e permanecem fora do
Git. Não há requeue automático: após `TIMEOUT`, consulte `sacct` e os logs e
submeta novamente o modo `resume`, que reutiliza o último checkpoint confirmado
ou reproduz a rodada incompleta.

## Investigar learning rate com canários completos

A calibração v4 é independente do piloto. Ela reutiliza os mesmos 20
perfis-canário disjuntos e treina quatro braços partindo do mesmo baseline.
Todos percorrem 160 vezes o bundle de 100 conversas; somente o learning rate
AdamW varia entre `1e-5`, `3e-5`, `1e-4` e `3e-4`. São 64.000 apresentações,
16.000 passos e 905 gerações greedy no total. O braço `1e-5` deve reproduzir o
fingerprint final da dose 160 da v3 antes que os demais resultados sejam
aceitos.

Valide primeiro, sem escrever:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_memorization_calibration \
  --config configs/memorization-calibration-v4.yaml \
  --device cuda \
  --preflight-only
```

No Slurm, use sempre um modo explícito:

```bash
sbatch scripts/run_learning_rate_calibration_l40s.sbatch preflight
sbatch scripts/run_learning_rate_calibration_l40s.sbatch start
sbatch scripts/run_learning_rate_calibration_l40s.sbatch resume
```

`start` cria `memorization-calibration-greedy-lr-seed-101-v4`; `resume` revalida
braços, checkpoints e auditorias já publicados. A execução oficial terminou com
`calibrated=true`, baseline reprovado e `lr-000030` como primeiro braço aprovado;
esse resultado foi integrado estritamente ao piloto v3. Consulte
[`docs/memorization-calibration.md`](docs/memorization-calibration.md).

## Calibrar a exposição local das vítimas no FedAvg

A execução `federated-memorization-calibration/v1` mede se aumentar somente o
treinamento local das vítimas torna a reprodução federada mensurável. Ela cria
três trajetórias F0 independentes, todas partindo do Tucano 2 pinado, com
multiplicadores 1×, 2× e 4×. O auxiliar benigno permanece com uma passagem,
25 passos e peso `1/11`; as dez vítimas mantêm peso `1/11` cada.

Somente B0 e o checkpoint final de cada braço são auditados, sempre com os 200
participantes e inferência greedy. O total fixado é de 146.000 apresentações,
36.500 passos, 7.204 gerações de extração e 2.000 avaliações de conversa de
utilidade. O braço 1× precisa reproduzir o modelo, a auditoria final e a
utilidade F0-r20 do piloto v3 concluído antes da interpretação dos braços 2×/4×.

Valide primeiro sem escrever:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_federated_memorization_calibration \
  --config configs/federated-memorization-calibration-v1.yaml \
  --device cuda \
  --preflight-only
```

No Slurm, use um único modo por submissão:

```bash
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch preflight
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch start
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch resume
```

`start` exige que o `run_id` oficial ainda não exista. `resume` revalida o
checkpoint confirmado e repete somente a rodada incompleta, pois o AdamW local
não é persistido. Consulte
[`docs/federated-memorization-calibration.md`](docs/federated-memorization-calibration.md).

## Executar a grade federada de intensidade

A grade v2 avalia `4×`, `8×` e `16×` com learning rates das vítimas `3e-5` e
`1e-4`, separadamente nas seeds `101` e `361506353`. O auxiliar permanece em
`3e-5`, 25 passos e peso `1/11`. Cada seed é um run retomável independente e os
dois jobs podem ocupar L40S diferentes em paralelo. Consulte
[`docs/federated-memorization-grid.md`](docs/federated-memorization-grid.md).

## Executar o piloto de substituição semântica

O piloto usa `1e-4 / 4×` somente nas vítimas, mantém o auxiliar em `3e-5 / 1×`
e executa F0, F1, F4 e F5 separadamente nas duas seeds. Em F4/F5, os nove
campos são substituídos localmente e mudam a cada rodada. Rode primeiro os dois
preflights:

```bash
sbatch --job-name=semantic-substitution-s101-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch preflight 101
sbatch --job-name=semantic-substitution-s361506353-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch preflight 361506353
```

Após os dois runs completos, gere o resumo conjunto com
`python -m federated_leakage.summarize_semantic_substitution_pilot --config
configs/semantic-substitution-pilot-v1.yaml`. O procedimento completo e as
regras de retomada estão em
[`docs/semantic-substitution-pilot.md`](docs/semantic-substitution-pilot.md).

## Executar o piloto refinado F0-F5

O piloto `refined-defense-pilot/v1` começa sempre do export Fórum/Tec original e
executa, por seed, F0/F1, F2/F3 com ε 3, F2/F3 com ε 8 e F4/F5. F0/F1 formam um
gate obrigatório nas duas seeds antes das defesas. Por isso, após ambos os jobs
produzirem seus gates, uma seed que terminou em `awaiting-peer-vulnerability-gate`
deve ser submetida novamente em modo `resume`.

Antes do piloto científico, a ordem operacional é:

```bash
# 1 e 100 passos privados reais, sem persistir pesos
python -m federated_leakage.smoke_private_training \
  --config configs/main-v5.yaml \
  --model-artifact-dir "$PWD/artifacts/models/ae3238fde6675942cac5" \
  --cache-dir artifacts/huggingface --device cuda --epsilon 3 --steps 1

python -m federated_leakage.smoke_private_training \
  --config configs/main-v5.yaml \
  --model-artifact-dir "$PWD/artifacts/models/ae3238fde6675942cac5" \
  --cache-dir artifacts/huggingface --device cuda --epsilon 3 --steps 100

# preflight de cada seed no Slurm
sbatch --job-name=refined-defense-s101-v1 \
  scripts/run_refined_defense_pilot_l40s.sbatch preflight 101
sbatch --job-name=refined-defense-s361506353-v1 \
  scripts/run_refined_defense_pilot_l40s.sbatch preflight 361506353
```

O launcher aceita apenas `preflight`, `start` ou `resume`, usa uma L40S por
seed, execução totalmente offline e `CUBLAS_WORKSPACE_CONFIG=:4096:8`. O resumo
conjunto é produzido somente após os dois `completed.json`:

```bash
python -m federated_leakage.summarize_refined_defense_pilot \
  --output-root outputs
```

Uma réplica operacional independente pode usar a RTX PRO 6000 Blackwell sem
alterar a `.venv`, o launcher L40S ou qualquer configuração científica. No
headnode, prepare uma única vez o ambiente CUDA 12.8:

```bash
scripts/prepare_rtxpro6000_cu128_env.sh
```

O preparador publica `.venv-rtxpro6000-cu128/` somente depois de validar
`torch==2.7.1+cu128`, `sm_120`, Opacus e `pip check`. Na fila Blackwell, use job
names próprios:

```bash
sbatch --job-name=refined-defense-rtxpro6000-s101-v1 \
  scripts/run_refined_defense_pilot_rtxpro6000.sbatch preflight 101
sbatch --job-name=refined-defense-rtxpro6000-s361506353-v1 \
  scripts/run_refined_defense_pilot_rtxpro6000.sbatch preflight 361506353
```

`start` e `resume` usam exclusivamente
`outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1/`; portanto não leem
nem sobrescrevem checkpoints ou resultados em `outputs/runs/`. Após concluir e
resumir separadamente as duas execuções, compare os hardwares sem combinar
médias:

```bash
python -m federated_leakage.summarize_refined_defense_pilot \
  --output-root outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1
python -m federated_leakage.summarize_refined_runtime_replication \
  --output-root outputs
```

Consulte [`docs/refined-dp-pilot.md`](docs/refined-dp-pilot.md) para gates,
checkpoints e critérios de classificação.

## Gerar um dataset para inspeção

Depois da instalação editável, uma única seed gera os dez clientes-vítima e as
20 rodadas auxiliares F0/F1 nas apresentações benigna e adversária:

```bash
python -m federated_leakage.generate_dataset --seed 11
```

O destino padrão é `outputs/datasets/inspection-seed-11-v4/`. A CLI aceita
opcionalmente `--dataset-id`, `--schedule-id`, `--output-root` e `--dry-run`,
recusa sobrescrita e só publica o bundle depois do preflight completo.

## Executar os testes

Depois de ativar um ambiente virtual:

```bash
python -m pip install -e '.[model]'
python -m unittest discover -s tests -v
```

Os testes cobrem a regressão v4 dos e-mails, a estabilidade v3 dos demais campos,
os datasets 10×100 das vítimas, o pareamento benigno/adversário, escopos de
perda, ordem canônica, anotações, colisões, round-trip JSONL, manifestos seguros,
tokenização, máscaras, padding, perda por conversa, gradient accumulation,
rollback, deltas em streaming, pesos FedAvg, aplicação atômica, pareamento F0/F1
e auditoria com orçamentos aninhados, checkpoints `safetensors`, retomada
transacional, continuidade independente das trajetórias e recuperação de falha
após auditoria, além do controle positivo canário com doses prefixadas,
checkpoint por braço e critério distintivo, substituição rotativa dos nove
campos, colisões falsas ambíguas, auditoria corrente/histórica, checkpoints do
piloto F0/F1/F4/F5 e carregamento estrito e offline do modelo. Os
testes com os pesos e o tokenizador reais são opt-in e exigem um cache já
preparado:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

FEDERATED_RUN_MODEL_SMOKE=1 \
python -m unittest tests.test_model_smoke -v

HF_HUB_OFFLINE=1 \
FEDERATED_RUN_TOKENIZATION_SMOKE=1 \
python -m unittest tests.test_tokenization_smoke -v

HF_HUB_OFFLINE=1 \
FEDERATED_RUN_TRAINING_SMOKE=1 \
python -m unittest tests.test_training_smoke -v

HF_HUB_OFFLINE=1 \
FEDERATED_RUN_AUDIT_SMOKE=1 \
python -m unittest tests.test_audit_smoke -v

HF_HUB_OFFLINE=1 \
FEDERATED_RUN_PILOT_PREFLIGHT_SMOKE=1 \
python -m unittest tests.test_pilot_smoke -v
```

O smoke de auditoria executa uma consulta de cada modo, não sela um resultado
científico e confirma que os parâmetros do modelo permanecem inalterados. O
smoke de treinamento executa somente um passo lógico real e restaura o modelo.
Ele não constitui uma atualização local válida nem grava pesos. O smoke do
piloto executa apenas o preflight real e não cria datasets, checkpoints nem
auditorias persistidas.

## Limites

Somente dados sintéticos são permitidos. Não versionar conjuntos de dados,
pesos, pontos de restauração, registros protegidos, mapas de substituição,
arquivos temporários ou saídas de execuções. Outro modelo entra apenas pelo
contrato de artefato e exige reexecutar toda a campanha.

O protocolo de DP-SGD prevê cada conversa como unidade. Como o mesmo registro
completo aparece em quatro conversas distintas, isso não autoriza alegação de
privacidade no nível do participante inteiro. Mudar a unidade de privacidade ou
contabilizar a contribuição completa exige recalcular e versionar o
contabilizador antes de executar a campanha.
