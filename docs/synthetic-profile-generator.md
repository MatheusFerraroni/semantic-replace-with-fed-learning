# Gerador de perfis sintéticos

## Objetivo

O gerador cria perfis inteiramente sintéticos e reproduzíveis sem gravar seus
valores no cliente auxiliar. A mesma entrada sempre reconstrói o mesmo perfil,
enquanto rodada ou índice diferentes produzem uma entidade diferente.

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

## Ciclo de uma rodada

1. O cliente recebe sua chave de fluxo e o número da rodada.
2. O gerador materializa 80 perfis e 20 conversas gerais em memória.
3. O validador verifica formato, checksum inválido, ordem, anotações e colisões.
4. O renderizador produz o prefixo e a continuação canônica.
5. O chamador tokeniza cada amostra uma vez e executa o treinamento local.
6. O cliente descarta o objeto da rodada; nenhum perfil bruto é serializado.

Uma rodada incompleta é sempre descartada. Na retomada, a mesma chave, versão,
rodada e configuração reconstroem exatamente os mesmos objetos.

## Campos

- `PERSON_NAME`: nome `pt_BR` com pseudossobrenome determinístico que impede
  repetição dentro do fluxo;
- `BIRTH_DATE`: data em uma faixa fixa e sem repetição dentro do fluxo;
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

Data e horário de atendimento podem se repetir separadamente ou como a mesma
combinação. Os demais tipos continuam sujeitos à verificação de colisão entre
todos os conjuntos experimentais.

## Persistência

O gerador não oferece exportação de perfis. O único artefato gravável por ele é
uma entrada de manifesto sem valores protegidos, contendo:

- versões do esquema, gerador e Faker;
- número da rodada e contagens;
- hash da ordem das entidades;
- hash do lote canônico;
- hash do template.

O arquivo `round_auxiliary_manifest.jsonl` fica no diretório externo da
execução, nunca no Git. A chave mestra ou sua referência protegida, a rodada
concluída e os hashes entram no ponto de restauração do executor confiável.

## Uso

```python
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    derive_stream_key,
)

auxiliary_key = derive_stream_key(
    trusted_master_key,
    experiment_seed=11,
    namespace="auxiliary",
    schedule_id="F0-F1",
)

generator = AuxiliaryRoundGenerator(auxiliary_key)
round_data = generator.generate(round_id=1)

for sample in round_data.profile_samples:
    use_in_training(sample.rendered.text)

# Depois do treinamento local:
del round_data
```

O exemplo de derivação é executado pelo componente confiável. O cliente recebe
`auxiliary_key`, não `trusted_master_key`.

## Validação confiável

Antes da campanha, o avaliador pode regenerar em memória todos os fluxos e
chamar o validador de coleção com os valores reservados dos outros conjuntos.
Qualquer colisão proibida interrompe a execução sem informar ao auxiliar qual
campo ou entidade colidiu. Somente os hashes da agenda aprovada precisam ser
preservados.

## Dependência e testes

O Faker é fixado em `40.36.0` porque suas saídas fazem parte do contrato de
reprodução. Os testes usam somente `unittest` da biblioteca padrão:

```bash
python -m unittest discover -s tests -v
```
