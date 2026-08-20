# Gerador de perfis e conversas sintéticas

## Objetivo

O gerador cria perfis e conversas inteiramente sintéticos e reproduzíveis. Os
perfis tipados e as chaves não são gravados; as conversas validadas podem ser
publicadas em JSONL no diretório atribuído ao cliente. A mesma entrada sempre
reconstrói o mesmo conjunto, enquanto rodada ou índice auxiliar diferentes
produzem uma entidade diferente. Os dez datasets das vítimas são estáveis entre
rodadas.

O código está em `src/federated_leakage/synthetic_profiles/`. Ele não depende de
dados reais, arquivos montados, bancos ou serviços externos.

## Separação das sementes

O executor confiável mantém uma chave mestra de pelo menos 256 bits fora do Git.
HMAC-SHA-256 deriva uma chave independente para vítimas, auxiliar, controles e
substituições. Somente a chave do fluxo auxiliar é entregue ao cliente auxiliar.

O fluxo auxiliar usa como contexto a semente experimental e o identificador do
par comparável. A derivação por perfil acrescenta rodada e índice; a derivação
por valor acrescenta o tipo do campo. O membro benigno/adversário do par e `k`
não participam da derivação. Por isso, F0/F1, F2/F3 e F4/F5 reconstroem os mesmos
valores em todos os pesos auxiliares.

Conhecer uma chave derivada não dá ao auxiliar a chave mestra nem as chaves dos
outros fluxos. A chave mestra e qualquer registro de valores das vítimas
continuam exclusivos do executor confiável e do avaliador.

## Contrato de conversa

`TrainingConversation` contém o texto ainda não tokenizado, IDs técnicos locais,
tipo `protected` ou `general`, anotações, template e escopo de perda. Para uma
conversa protegida, `prefix_length` mede caracteres Unicode até o fim do prefixo;
o futuro treinador será responsável por convertê-lo em máscara de tokens depois
de tokenizar a amostra completa uma única vez. A ordem da tupla de conversas é a
agenda determinística; `sample_index` identifica a posição lógica anterior à
permutação e não deve ser usado para reordenar o dataset.

O catálogo `training-conversation-catalog/v1` é definido em
`conversations.py`. As quatro molduras naturais começam pelo segmento canônico e
acrescentam somente `\nASSISTENTE: ` e uma resposta neutra. As 20 conversas
gerais são aceitas apenas por igualdade literal com o catálogo, não possuem
anotações e usam perda integral.

## Datasets das vítimas

`VictimDatasetGenerator` recebe apenas a chave do fluxo vítima e gera dez
`VictimClientDataset`. Cada cliente contém 20 entidades e 100 conversas: quatro
protegidas e uma geral por entidade. A ordem é uma permutação HMAC estável e a
derivação não inclui rodada, cenário nem `k`.

## Ciclo de uma rodada auxiliar

1. O cliente recebe sua chave de fluxo e o número da rodada.
2. O gerador materializa 80 perfis e 20 conversas gerais em memória.
3. A apresentação benigna envolve os perfis com as quatro molduras naturais; a
   adversária mantém somente o segmento canônico.
4. O validador verifica formato, checksum inválido, ordem, anotações, perda,
   pareamento e colisões.
5. O cliente pode publicar as 100 conversas em seu próprio JSONL validado.
6. O chamador carrega o arquivo, tokeniza cada amostra uma vez e executa o
   treinamento local.
7. O cliente descarta o perfil tipado; nenhuma chave é serializada.

Nos 80 registros adversários, a perda começa depois do prefixo e cobre a
continuação canônica completa. As 20 conversas gerais usam perda integral nas
duas apresentações.

Uma rodada incompleta é sempre descartada. Na retomada, a mesma chave, versão,
rodada e configuração reconstroem exatamente os mesmos objetos.

## Campos

- `PERSON_NAME`: nome `pt_BR` com pseudossobrenome determinístico que impede
  repetição dentro do fluxo;
- `BIRTH_DATE`: data entre `1966-01-01` e `2006-12-31`, equivalente a 20–60
  anos na referência fixa `2026-12-31`, com repetição permitida;
- `CPF`: aparência preservada e segundo dígito verificador deliberadamente
  incorreto;
- `RG`: formato de nove posições e dígito deliberadamente incompatível com a
  convenção sintética adotada;
- `PHONE`: DDD inválido `00`;
- `EMAIL`: domínio reservado `synthetic.invalid`;
- `ADDRESS`: cidade, UF e CEP explicitamente sintéticos;
- `APPOINTMENT_DATE`: data entre 2026 e 2027, com repetição permitida;
- `APPOINTMENT_TIME`: horário entre `08:00` e `18:45`, em intervalos de 15
  minutos, com repetição permitida.

Data de nascimento, data e horário de atendimento podem se repetir separadamente
ou em combinação. Nome, CPF, RG, telefone, e-mail e endereço continuam sujeitos
à verificação de colisão entre todos os conjuntos experimentais.

## Persistência

`storage.py` publica uma conversa por linha, em JSON canônico UTF-8 com `LF`, em:

```text
outputs/datasets/<dataset_id>/
├── trusted/manifests/
└── clients/
    ├── victim/<client_id>/conversations.jsonl
    └── auxiliary/<schedule_id>/<presentation>/round-001/conversations.jsonl
```

Cada JSONL possui `metadata.json` adjacente com versões, identidade lógica,
contagem e SHA-256 dos bytes. Escrita é exclusiva e atômica; leitura rejeita
campos desconhecidos, conteúdo adulterado, identidade divergente e caminhos
inseguros. O bundle deve ficar em `outputs/`, já ignorado pelo Git. O chamador
entrega a cada componente somente seu caminho; a raiz não é compartilhada com o
servidor nem com o auxiliar adversário.

Os manifestos continuam sem conteúdo protegido e contêm:

- versões do esquema, gerador e Faker;
- número da rodada, apresentação e contagens auxiliares;
- hashes separados de agenda, valores, apresentação e lote;
- hashes agregados e por cliente dos datasets das vítimas;
- hashes do template canônico e do catálogo.

`victim_dataset_manifest.json` fica em `trusted/manifests/`; o manifesto auxiliar
continua associado à execução da rodada. Nenhum manifesto ou metadado contém
texto, valor protegido, anotação ou `entity_id`. Chaves mestras e de fluxo nunca
entram no bundle.

## Uso

```python
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
    derive_stream_key,
    read_auxiliary_round,
    read_victim_client_dataset,
    write_auxiliary_round,
    write_victim_datasets,
)

from pathlib import Path

auxiliary_key = derive_stream_key(
    trusted_master_key,
    experiment_seed=11,
    namespace="auxiliary",
    schedule_id="F0-F1",
)

generator = AuxiliaryRoundGenerator(auxiliary_key)
benign_round = generator.generate(round_id=1, presentation="benign")
adversarial_round = generator.generate(round_id=1, presentation="adversarial")

victim_key = derive_stream_key(
    trusted_master_key,
    experiment_seed=11,
    namespace="victim",
    schedule_id="victims",
)
victim_datasets = VictimDatasetGenerator(victim_key).generate()

output_root = Path("outputs/datasets")
dataset_id = "seed-11-main-v2"
write_victim_datasets(output_root, dataset_id, victim_datasets)
write_auxiliary_round(output_root, dataset_id, "F0-F1", benign_round)

# Cada consumidor abre somente seu próprio caminho lógico.
victim_01 = read_victim_client_dataset(
    output_root, dataset_id, "victim-01"
)
auxiliary_round_01 = read_auxiliary_round(
    output_root, dataset_id, "F0-F1", "benign", 1
)

# Depois do treinamento local, descarte os objetos em memória.
del benign_round, adversarial_round, victim_datasets
```

O exemplo de derivação é executado pelo componente confiável. O cliente recebe
`auxiliary_key`, não `trusted_master_key`.

## Validação confiável

Antes da campanha, o avaliador pode regenerar em memória todos os fluxos e usar
`validate_conversation_preflight` com os dez datasets das vítimas, as 20 rodadas
de uma agenda auxiliar pareada e valores reservados dos conjuntos futuros.
Qualquer colisão proibida interrompe a execução com uma mensagem genérica, sem
informar campo, valor ou entidade. Cada agenda pareada é verificada separadamente.
Não há remapeamento silencioso; somente os hashes da agenda aprovada precisam ser
preservados.

## Dependência e testes

O Faker é fixado em `40.36.0` porque suas saídas fazem parte do contrato de
reprodução. Os testes usam somente `unittest` da biblioteca padrão:

```bash
python -m unittest discover -s tests -v
```
