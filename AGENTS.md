# AGENTS.md

## Objetivo

Construir uma implementação de pesquisa pequena e reprodutível para avaliar a
reprodução indevida de dados pessoais sintéticos durante treinamento federado de
todos os parâmetros do Tucano 2 0.6B.

## Limites do projeto

- Usar somente perfis, conversas, dados pessoais e canários sintéticos.
- Nunca adicionar dados reais, caminhos de volumes montados, ingestão de corpus
  ou pré-treinamento continuado a este projeto.
- Aceitar o modelo original ou um artefato externo compatível do Hugging Face
  somente pelo contrato de artefato documentado.
- Não versionar conjuntos de dados, pesos, registros protegidos, mapas de
  substituição, arquivos temporários, pontos de restauração ou saídas geradas
  pelas execuções.

## Invariantes experimentais

- Usar por padrão `Polygl0t/Tucano2-0.6B-Base` na revisão
  `dad97dc864a8f9a1d240fb9351d098f3af9511d7` e rotular essas execuções como
  `upstream_baseline`.
- Manter inalterados o tokenizador, o vocabulário e os tokens especiais.
- Usar 1.024 como comprimento máximo de treinamento, manter as amostras curtas e
  treinar todos os parâmetros em todas as condições federadas.
- Manter 10 clientes-vítima e um slot auxiliar em cada cenário federado.
- Distinguir cliente federado de participante sintético: cada cliente-vítima
  possui 20 participantes sintéticos, totalizando 200 participantes-vítima.
- Tratar todos os nove campos do perfil como dados pessoais protegidos: nome,
  data de nascimento, CPF, RG, telefone, e-mail, endereço, data e horário de
  atendimento.
- Manter cinco conversas por participante-vítima: quatro devem conter o registro
  canônico completo com os mesmos nove valores protegidos da entidade e uma deve
  ser geral, sem qualquer dado ou fato individualizado do participante.
- Fixar a sequência canônica do registro em `PERSON_NAME`, `BIRTH_DATE`, `CPF`,
  `RG`, `PHONE`, `EMAIL`, `ADDRESS`, `APPOINTMENT_DATE` e `APPOINTMENT_TIME`,
  nessa ordem. Não permitir reordenação, omissão, repetição nem mudança dos
  rótulos ou delimitadores na campanha principal.
- Usar exatamente o mesmo prefixo, template de continuação e ordem canônica nos
  dados das vítimas, nas variantes auxiliares, nas substituições e na auditoria.
  Variação de ordem pertence somente a uma ablação futura.
- Não fornecer ao adversário nenhum nome, campo, perfil, instrução de auditoria
  ou resultado pertencente às vítimas.
- Fora do respectivo cliente-vítima, permitir que somente o avaliador confiável
  acesse os nomes e o registro de respostas corretas. O nome não entra no
  denominador da extração direcionada porque o próprio avaliador o insere na
  instrução. Em consultas sem nome, sua reprodução exata conta como exposição.
- Não permitir dados nem fatos individualizados na única conversa geral de cada
  perfil. Se algum for introduzido, ele deve ser registrado, protegido e
  auditado como os demais dados do participante.
- Preservar a aparência dos documentos sintéticos, forçar checksums inválidos e
  rejeitar colisões de nome, CPF, RG, telefone, e-mail e endereço entre vítimas,
  auxiliar, controles e substituições. Data de nascimento, data de atendimento
  e horário de atendimento podem se repetir entre entidades. Gerar nascimentos
  entre `1966-01-01` e `2006-12-31`, equivalentes a 20–60 anos na referência
  fixa `2026-12-31`.
- Gerar horários de atendimento somente em intervalos humanos de 15 minutos,
  com minutos `00`, `15`, `30` ou `45`.
- Usar extração condicionada ao nome fornecido pelo avaliador, com continuação do
  perfil completo, como objetivo principal. Auditar também cada tipo de campo
  com instruções específicas e executar extração sem nome como controle
  complementar.
- Usar cliente auxiliar benigno em F0, F2 e F4 e a variante adversária pareada em
  F1, F3 e F5.
- Parear cada comparação benigna/adversária pelo modelo inicial, dados das
  vítimas, especificação e agenda auxiliar por rodada, valores, ordem das
  amostras, coeficiente FedAvg e passos locais.
- Fazer cada variante auxiliar reconstruir localmente a mesma agenda
  determinística da comparação, sem compartilhar arquivos ou estado privado.
- Fazer o cliente adversário gerar localmente dados auxiliares sintéticos novos
  no início de cada rodada. Não reutilizar perfis, nomes nem os demais valores
  protegidos sujeitos à unicidade entre amostras ou rodadas; data de nascimento,
  data e horário de atendimento são as exceções e podem se repetir.
- Na referência, usar renovação por rodada não adaptativa: a política e a
  derivação de sementes são fixadas antes da execução, mas os dados são gerados
  dentro do cliente a cada rodada e não dependem das respostas do modelo global.
  Geração condicionada ao modelo pertence à ablação adaptativa.
- Durante seu treinamento, o adversário usa apenas nomes e valores auxiliares
  gerados por ele. Nenhum nome ou outro valor de vítima pode entrar em seu
  processo. Somente o avaliador usa nomes de vítimas nas consultas posteriores.
- Permitir que o cliente adversário controle seus dados, função de perda,
  otimizador, passos locais e atualização submetida. Ele recebe somente o modelo
  global de cada rodada e nunca recebe atualizações individuais das vítimas.
- Usar a mesma receita local em todo `k`: continuações positivas, treinamento
  somente na continuação e nenhuma transformação da atualização submetida. Em
  `k=1`, o coeficiente FedAvg é `1/11`; nos demais pontos, somente os coeficientes
  normalizados mudam. Tratar mistura positiva/negativa, outras capacidades,
  adaptatividade e escala do delta como ablações separadas.
- Tratar a massa de agregação auxiliar como dimensão da campanha principal em
  todos os cenários F0-F5. Manter um slot auxiliar físico e ponderá-lo com
  `k = 1..10` unidades virtuais: participação auxiliar `k/(10+k)` e participação
  de cada vítima `1/(10+k)`.
- Parear F0/F1, F2/F3 em cada valor de ε e F4/F5 em todo `k`, parar na divisão
  50/50, relatar todos os pontos e não confundir a varredura com escala do delta
  submetido. B0 não possui `k`, pois não executa agregação federada.
- Tokenizar cada amostra de ataque uma vez. Mascarar os IDs exatos do prefixo e
  calcular a perda sobre a continuação canônica completa, como média por conversa
  seguida da média do lote lógico.
- Aplicar DP-SGD ou substituição semântica somente aos 10 clientes-vítima.
- Usar cada conversa como unidade de privacidade do DP-SGD. Como o mesmo registro
  completo aparece em quatro unidades de privacidade distintas, não alegar DP no
  nível do participante enquanto as cinco conversas não forem agregadas como uma
  única unidade protegida ou a contribuição completa não for contabilizada por
  composição de grupo explicitamente versionada.
- Recalcular e versionar os parâmetros do contabilizador de privacidade antes da
  campanha sempre que mudarem a unidade de privacidade, amostragem, lote, passos
  ou rodadas. Não reutilizar sigmas de uma configuração incompatível.
- Manter substituições estáveis por entidade e tipo e idênticas no par F4/F5.
  Aplicar a mesma substituição às quatro conversas protegidas da entidade.
  O nome permanece no conjunto transformado somente para que o avaliador consiga
  aplicar o mesmo gatilho nos pares comparáveis. O adversário não recebe esse
  nome, e a exceção não o reclassifica como dado não protegido.
- Manter os conjuntos das vítimas disjuntos. O adversário nunca pode acessar
  dados, valores protegidos, atualizações locais, arquivos do auditor ou mapas de
  substituição das vítimas.
- Manter o avaliador confiável separado de todos os clientes e do servidor. Ele
  é o único componente que conhece os nomes das vítimas, constrói as instruções
  com esses nomes e acessa o registro completo para pontuação. O adversário não
  possui capacidade de consulta direcionada nesta versão.
- Não compartilhar com o adversário as instruções, gerações, métricas ou
  resultados produzidos pelo avaliador.
- Manter o servidor honesto e limitado ao FedAvg simples. Ele não compartilha
  atualizações locais e não usa agregação segura nem robusta neste protocolo.
- Auditar o modelo inicial e o modelo global após cada rodada, com instruções e
  sementes de geração fixas, sem alterar o treinamento posterior.
- Usar, na rodada 20, a reprodução exata de pares corretos
  `nome fornecido pelo avaliador -> tipo -> valor protegido` como métrica
  principal. Relatar também perfis completos, participantes com qualquer campo
  exposto, perfis completos na ordem canônica, resultados por tipo, associações
  incorretas e exposições sem nome.
- Tratar a trajetória das rodadas como medidas descritivas repetidas, não como
  execuções independentes.
- Manter os artefatos de auditoria com nomes, registros protegidos e mapas de
  substituição exclusivos do avaliador e fora do controle de versão. Relatar
  reproduções auxiliares apenas como
  diagnóstico de aprendizado do gatilho e sobreajuste.
- Executar o piloto de desenvolvimento documentado antes de congelar a receita
  principal. Não incluir sua semente nos resultados principais.
- Reiniciar e reexecutar todos os cenários quando o artefato inicial mudar.

## Regras de implementação

- Manter a implementação simples, com Python direto e módulos pequenos e
  reutilizáveis.
- Não adicionar estruturas, serviços, bancos, painéis ou abstrações que o
  protocolo não exija.
- Antes de adicionar uma dependência, verificar se a pilha existente atende à
  necessidade.
- Tornar as execuções reprodutíveis com sementes fixas e configurações
  versionadas.
- Interromper a execução quando falharem verificações de isolamento, anotação,
  colisão, geração por rodada ou acesso a dados protegidos.
- Salvar métricas separadamente dos pontos de restauração e evitar pontos de
  restauração ou intermediários grandes sem necessidade.
- Preferir rotinas retomáveis para geração, treinamento e auditoria demorados.
- Não alterar pressupostos experimentais silenciosamente. Registrar mudanças no
  protocolo, README ou configuração da execução.
- Relatar privacidade e utilidade juntas, inclusive resultados negativos e
  inconclusivos.

Otimizar para clareza científica e reprodutibilidade, não para infraestrutura de
produção.
