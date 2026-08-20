# Gerador de perfis e conversas sintéticas

## Objetivo

O gerador cria perfis e conversas inteiramente sintéticos e reproduzíveis. Os
perfis tipados e o estado interno de derivação não são gravados; as conversas
validadas podem ser publicadas em JSONL no diretório atribuído ao cliente. A mesma entrada sempre
reconstrói o mesmo conjunto, enquanto rodada ou índice auxiliar diferentes
produzem uma entidade diferente. Os dez datasets das vítimas são estáveis entre
rodadas.

O código está em `src/federated_leakage/synthetic_profiles/`. Ele não depende de
dados reais, arquivos montados, bancos ou serviços externos.

## Derivação da seed

Uma única seed inteira e não negativa controla toda a geração. SHA-256 deriva
material determinístico separado por rótulos versionados para vítimas, auxiliar,
agenda, rodada, amostra e campo.

O fluxo auxiliar usa como contexto a seed experimental e o identificador do
par comparável. A derivação por perfil acrescenta rodada e índice; a derivação
por valor acrescenta o tipo do campo. O membro benigno/adversário do par e `k`
não participam da derivação. Por isso, F0/F1, F2/F3 e F4/F5 reconstroem os mesmos
valores em todos os pesos auxiliares.

A seed não é segredo e a derivação não oferece isolamento criptográfico. Os
rótulos impedem apenas acoplamento acidental das sequências. O isolamento do
experimento depende da orquestração: cada componente recebe somente sua API e
seu caminho local, nunca objetos ou arquivos de outro papel.

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

`VictimDatasetGenerator` recebe a seed e gera dez
`VictimClientDataset`. Cada cliente contém 20 entidades e 100 conversas: quatro
protegidas e uma geral por entidade. A ordem é uma permutação determinística
estável, e a derivação não inclui rodada, cenário nem `k`.

## Ciclo de uma rodada auxiliar

1. O cliente recebe a seed, a agenda e o número da rodada.
2. O gerador materializa 80 perfis e 20 conversas gerais em memória.
3. A apresentação benigna envolve os perfis com as quatro molduras naturais; a
   adversária mantém somente o segmento canônico.
4. O validador verifica formato, checksum inválido, ordem, anotações, perda,
   pareamento e colisões.
5. O cliente pode publicar as 100 conversas em seu próprio JSONL validado.
6. O chamador carrega o arquivo, tokeniza cada amostra uma vez e executa o
   treinamento local.
7. O cliente descarta o perfil tipado; nenhum estado interno é serializado.

Nos 80 registros adversários, a perda começa depois do prefixo e cobre a
continuação canônica completa. As 20 conversas gerais usam perda integral nas
duas apresentações.

Uma rodada incompleta é sempre descartada. Na retomada, a mesma seed, versão,
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
- `EMAIL`: variação ASCII determinística do primeiro nome, sobrenome e
  pseudossobrenome sintético, opcionalmente com ano de nascimento, usando um dos
  domínios `gmail.com`, `outlook.com`, `hotmail.com`, `yahoo.com`, `icloud.com`
  ou `proton.me`;
- `ADDRESS`: cidade, UF e CEP explicitamente sintéticos;
- `APPOINTMENT_DATE`: data entre 2026 e 2027, com repetição permitida;
- `APPOINTMENT_TIME`: horário entre `08:00` e `18:45`, em intervalos de 15
  minutos, com repetição permitida.

Data de nascimento, data e horário de atendimento podem se repetir separadamente
ou em combinação. Nome, CPF, RG, telefone, e-mail e endereço continuam sujeitos
à verificação de colisão entre todos os conjuntos experimentais.

Os domínios de e-mail são reais por decisão do protocolo. Embora o local part
seja derivado somente de perfis sintéticos, não há garantia de inexistência ou
não roteamento do endereço completo, e nenhum deles deve ser contatado.

## Persistência

`storage.py` publica uma conversa por linha, em JSON canônico UTF-8 com `LF`, em:

```text
outputs/datasets/<dataset_id>/
├── trusted/manifests/
│   ├── generation_manifest.json
│   ├── round_auxiliary_manifest.jsonl
│   └── victim_dataset_manifest.json
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
- hashes do template canônico e do catálogo;
- seed, agenda, contagens e hashes agregados do bundle completo de inspeção.

Os manifestos ficam em `trusted/manifests/`. Nenhum manifesto ou metadado contém
texto, valor protegido, anotação ou `entity_id`. O estado interno da derivação
nunca entra no bundle.

## Uso da CLI

O comando abaixo executa os dois preflights, valida os 20 pares auxiliares e
publica atomicamente 5.000 conversas:

```bash
python -m federated_leakage.generate_dataset --seed 11
```

O destino padrão é `outputs/datasets/inspection-seed-11-v4/`. Use `--dry-run`
para validar tudo em memória. `--dataset-id`, `--schedule-id` e `--output-root`
são opcionais; nenhum destino existente é sobrescrito.

## Uso da API

```python
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
    read_auxiliary_round,
    read_victim_client_dataset,
    write_auxiliary_round,
    write_victim_datasets,
)

from pathlib import Path

generator = AuxiliaryRoundGenerator(11, schedule_id="F0-F1")
benign_round = generator.generate(round_id=1, presentation="benign")
adversarial_round = generator.generate(round_id=1, presentation="adversarial")

victim_datasets = VictimDatasetGenerator(11).generate()

output_root = Path("outputs/datasets")
dataset_id = "seed-11-main-v4"
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

Na campanha, cada componente recebe somente a API e o caminho correspondentes ao
seu papel. A seed, por si só, não é um mecanismo de controle de acesso.

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
