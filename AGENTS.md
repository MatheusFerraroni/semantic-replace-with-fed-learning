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
- Antes de iniciar qualquer processo Python com CUDA determinístico, exportar
  exatamente `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A execução deve falhar sem
  corrigir silenciosamente a variável quando ela estiver ausente ou divergente.
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
  rejeitar colisões de nome, CPF, RG, telefone, e-mail e endereço entre os fluxos
  originais de vítimas, auxiliar e controles. Substitutos desses tipos não podem
  coincidir com originais, mas colisões entre valores falsos são permitidas e
  auditadas. Data de nascimento, data de atendimento e horário de atendimento
  podem se repetir entre entidades. Gerar nascimentos
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
- Parear cada comparação benigna/adversária pelo mesmo baseline no início das
  duas trajetórias, dados das vítimas, especificação e agenda auxiliar por
  rodada, valores, ordem das amostras, coeficiente FedAvg e passos locais. Na
  rodada 1, os dois estados iniciais devem coincidir com o baseline. Da rodada
  2 em diante, cada trajetória começa de seu próprio estado final anterior;
  não exigir igualdade entre os estados globais correntes de braços que já
  divergiram.
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
- No piloto refinado v1, usar DP-AdamW somente nas vítimas, com Opacus `1.6.0`,
  conversa como unidade, Poisson `q=0,04`, 100 passos por rodada, 2.000 passos
  totais, clipping flat `1,0`, lote físico máximo 1, `delta=1e-5`, hooks e
  `secure_mode=false`. Fixar sigma `2,81` para ε 3 e `1,36` para ε 8 e falhar
  se o accountant não reproduzir ε `2,98777705562`/ordem `7,4` ou ε
  `7,96431428079`/ordem `3,7`.
- Manter um accountant por cliente-vítima durante as 20 rodadas e relatar o
  máximo dos dez ε, nunca a soma. Reiniciar o AdamW a cada cliente e rodada.
  F2/F3 do mesmo orçamento devem reconstruir as mesmas agendas Poisson e de
  ruído; seed e ε diferentes usam fluxos separados. O auxiliar nunca recebe DP.
- Não persistir nem publicar loss, normas, clipping rate ou métricas individuais
  das vítimas privadas. Checkpoints podem conter somente o modelo global, os
  estados mínimos dos accountants, ε, fingerprints e hashes seguros.
- No piloto rotativo v1, substituir os nove campos, inclusive o nome, antes da
  tokenização. Manter o mapa estável nas quatro conversas e quatro passagens
  locais da entidade, renová-lo na rodada federada seguinte e reconstruir a
  mesma agenda em F4/F5 sem compartilhar estado privado.
- Fixar a agenda de substituição em `rotating-profile/v3`. Alterar seu domínio
  exige nova versão e novo preflight conjunto; não fornecer valores globais aos
  clientes para corrigir colisões em tempo de execução.
- Proibir a reutilização do próprio original e de substitutos anteriores da
  entidade. Para nome, CPF, RG, telefone, e-mail e endereço, proibir também
  coincidência com qualquer original validado. Permitir colisões entre valores
  falsos, registrando multiplicidades e tratando aliases compartilhados como
  ambíguos na auditoria.
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
- Auditar o modelo inicial e o modelo global após cada rodada com inferência
  exclusivamente greedy token a token: `do_sample=false`, um beam, uma sequência,
  penalidade de repetição 1 e cache habilitado. Não enviar `temperature`, `top_p`
  ou `top_k`, não semear a geração e não consumir RNG. Greedy significa argmax
  condicional por passo, não a sequência globalmente mais provável.
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
- Executar o piloto retomável com seed `101`, `k=1`, B0 compartilhado, F0 antes
  de F1 e 20 rodadas por trajetória. Auditar 20 alvos em B0 e após cada rodada;
  auditar também 1, 5 e 200 alvos em B0 e na rodada 20 de F0/F1. Não congelar a
  receita principal sem revisão humana explícita.
- Manter artefatos sampling v1 e as calibrações greedy v2/v3 imutáveis e
  somente para inspeção histórica. O runner da investigação de learning rate
  v4 não pode retomar, migrar nem combinar configurações, journals,
  checkpoints ou diretórios anteriores.
- Ao retomar o piloto, validar a continuidade de cada prefixo confirmado e de
  cada checkpoint: rodada 1 parte do baseline compartilhado e toda rodada
  posterior parte do estado final anterior do mesmo cenário.
- Manter checkpoints federados permanentes nas rodadas 1, 10 e 20 e somente um
  checkpoint móvel nas demais. Persistir pesos exclusivamente em `safetensors`,
  sem otimizadores, deltas, tokens, textos ou registros protegidos.
- Reiniciar e reexecutar todos os cenários quando o artefato inicial mudar.
- O piloto refinado usa exclusivamente o perfil `queroquero-export-v1` do
  artefato Fórum/Tec `ae3238fde6675942cac5`, braço `forum_tech` com 52.000
  passos. Validar ZIP, manifesto, inventário, pesos FP32 e equivalência integral
  do tokenizador; carregar diretamente em BF16 com atenção eager e sem fallback.
- Executar F0/F1 antes de F2-F5 no piloto refinado e exigir nas duas seeds pelo
  menos 50 pares distintivos, 25 vítimas e dois tipos distintivos, com B0 abaixo
  do gate. Se qualquer seed falhar, encerrar como inconclusivo sem executar
  silenciosamente as defesas.
- Investigar a capacidade de memorização em execução separada com seed `101`,
  20 perfis-canário completos e disjuntos e quatro braços independentes, todos
  com 160 repetições e partindo do mesmo baseline pinado. Variar somente o
  learning rate AdamW entre `1e-5`, `3e-5`, `1e-4` e `3e-4`.
- Manter um único AdamW dentro de cada braço canário e reiniciá-lo entre braços.
  O learning rate não entra na derivação da seed nem na ordem das 100
  conversas. O braço `1e-5` deve reproduzir exatamente o fingerprint final da
  dose 160 da calibração greedy v3 no mesmo dispositivo, versões e ambiente.
- Auditar o baseline e os quatro braços canários com os mesmos 20 alvos e
  prompts greedy. Considerar o controle calibrado somente com pelo menos 10 pares
  distintivos exatos distribuídos por pelo menos cinco canários; executar todas
  as taxas e registrar resultado negativo sem escalada automática.
- A investigação v4 selecionou `3e-5` como o menor learning rate testado que
  atingiu o gate: 100/100 pares distintivos e 20/20 canários, com o baseline
  reprovado. O piloto promovido usa essa taxa igualmente nos 10 clientes-vítima
  e no auxiliar, mas sempre começa do Tucano 2 pinado, nunca do checkpoint
  canário.
- Manter dados, checkpoints e artefatos privados da calibração separados do
  piloto e da campanha. O piloto greedy exige um braço aprovado, o baseline
  reprovado e uma integração versionada explícita do gate vigente.
  `calibrated=false` ou `baseline_gate_passed=true` bloqueia o piloto e o
  desenvolvimento das defesas até decisão explícita de protocolo.
- Avaliar utilidade sintética held-out no piloto promovido em B0, F0 rodada 20
  e F1 rodada 20, usando 100 perfis e 500 conversas exclusivas do fluxo de
  utilidade. Relatar perda média por conversa, NLL por token e perplexidade sem
  limiar automático; a promoção da receita continua sujeita a revisão humana.
- Calibrar a exposição local federada em execução separada com seed `101`, F0,
  `k=1`, AdamW `3e-5` e três braços independentes de 20 rodadas iniciados no
  baseline pinado. Repetir somente as 100 conversas de cada vítima em `1×`, `2×`
  ou `4×`; manter o auxiliar benigno com uma passagem, 25 passos e peso `1/11`.
- Na calibração federada, manter um AdamW por vítima e rodada durante suas
  repetições, reiniciá-lo entre clientes e não incluir o multiplicador na seed
  nem na ordem. O executor oficial fora desse contrato continua exigindo 25
  passos por cliente.
- Auditar B0 e os três endpoints da calibração federada com 200 alvos. Exigir
  pelo menos 10 pares distintivos exatos distribuídos por cinco vítimas e B0
  reprovado; executar todos os braços e registrar resultado negativo sem escalada
  automática. Avaliar também as mesmas 500 conversas held-out nos quatro modelos.
- Tratar o braço federado `1×` como regressão do F0-r20 do piloto v3 concluído.
  Validar o marcador histórico seguro e exigir modelo final, auditoria greedy de
  200 alvos e utilidade idênticos, sem reutilizar pesos, dados privados nem
  checkpoints do piloto.
- Executar a grade federada v2 separadamente nas seeds `101` e `361506353`, com
  seis braços F0 independentes por seed: vítimas em `3e-5` ou `1e-4` e
  repetições `4×`, `8×` ou `16×`. O auxiliar permanece em `3e-5`, 25 passos e
  peso `1/11`; LR e multiplicador não entram na seed nem na ordem local.
- Antes de cada run da grade, reconstruir ambas as seeds e validar colisões
  conjuntamente entre vítimas, agendas auxiliares, canários e utilidade. Auditar
  B0 e os seis endpoints com 200 alvos e avaliar as mesmas 500 conversas
  held-out por modelo.
- Considerar um braço intenso aprovado em uma seed somente com B0 reprovado,
  pelo menos 50 pares distintivos exatos, 25 vítimas expostas e dois tipos
  distintivos atingidos. Classificar o par de seeds como `robust`, `unstable`
  ou `insufficient`, sem promoção automática e sem mascarar seeds por média.
- Exigir que `4×/3e-5` na seed 101 reproduza modelo, auditoria e utilidade da
  calibração federada v1 concluída. Runs, checkpoints e schemas v1 não podem ser
  retomados ou migrados pelo runner v2.

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
- Executar o piloto em uma única L40S pelo launcher
  `scripts/run_pilot_lr_000030_l40s.sbatch`, sempre com modo explícito
  `preflight`, `start` ou `resume`. O launcher anterior permanece histórico.
  Usar dependência Slurm `singleton`, manter os logs na raiz ignorada do projeto
  e retomar manualmente após `TIMEOUT`; não configurar requeue automático.
- Executar a investigação de learning rate em uma única L40S pelo launcher
  `scripts/run_learning_rate_calibration_l40s.sbatch`, também com modo
  explícito, ambiente offline, dependência `singleton` e retomada manual. O
  launcher anterior da calibração v3 permanece somente para histórico.
- Executar a calibração federada de exposição em uma única L40S pelo launcher
  `scripts/run_federated_memorization_calibration_l40s.sbatch`, com modo
  explícito, 24 horas, ambiente offline, dependência `singleton` e retomada
  manual. Uma rodada incompleta deve ser repetida; não persistir AdamW.
- Executar cada seed da grade federada pelo launcher
  `scripts/run_federated_memorization_grid_l40s.sbatch`, com job name específico
  por seed. `singleton` impede duplicatas da mesma seed, mas permite duas L40S
  em paralelo. Usar modos explícitos, 24 horas, ambiente offline e retomada
  manual; não persistir AdamW nem configurar requeue.
- Executar cada seed do piloto de substituição pelo launcher
  `scripts/run_semantic_substitution_pilot_l40s.sbatch`, com job name específico
  por seed, modos `preflight`, `start` ou `resume`, ambiente offline e retomada
  manual. Nunca persistir conversas substituídas nem mapas nos clientes.
- Executar cada seed do piloto refinado pelo launcher
  `scripts/run_refined_defense_pilot_l40s.sbatch`, com job name específico por
  seed, modos `preflight`, `start` ou `resume`, uma L40S, ambiente offline,
  Opacus `1.6.0` e retomada manual. Executar primeiro smokes reais de 1 e 100
  passos privados; incompatibilidade de hooks ou OOM deve falhar sem fallback.
- Tratar a RTX PRO 6000 Blackwell somente como réplica de hardware independente
  da L40S. Usar `.venv-rtxpro6000-cu128`, PyTorch `2.7.1+cu128`, perfil
  `execution-runtime-profile/v1` e o launcher
  `scripts/run_refined_defense_pilot_rtxpro6000.sbatch`. Manter todos os seus
  manifestos, checkpoints, auditorias e resultados em
  `outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1/`; nunca retomar ou
  sobrescrever `outputs/runs/` e nunca combinar médias entre hardwares.
- Não alterar pressupostos experimentais silenciosamente. Registrar mudanças no
  protocolo, README ou configuração da execução.
- Relatar privacidade e utilidade juntas, inclusive resultados negativos e
  inconclusivos.

Otimizar para clareza científica e reprodutibilidade, não para infraestrutura de
produção.
