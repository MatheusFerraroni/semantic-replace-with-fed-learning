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
- Considerar o nome conhecimento auxiliar prévio do adversário. Ele continua
  protegido, mas não entra no denominador da extração direcionada porque já é
  fornecido na instrução. Em consultas sem nome, sua reprodução exata conta como
  exposição.
- Não permitir outros fatos individualizados nas quatro conversas gerais de
  cada perfil. Se algum for introduzido, ele deve ser registrado, protegido e
  auditado como os demais dados do participante.
- Preservar a aparência dos documentos sintéticos, forçar checksums inválidos e
  rejeitar colisões entre vítimas, auxiliar, controles e substituições.
- Usar extração direcionada pelo nome conhecido, com continuação do perfil
  completo, como objetivo principal. Auditar também cada tipo de campo com
  instruções específicas e executar extração sem nome como controle complementar.
- Usar cliente auxiliar benigno em F0, F2 e F4 e a variante adversária pareada em
  F1, F3 e F5.
- Parear cada comparação benigna/adversária pelo modelo inicial, dados das
  vítimas, especificação e agenda auxiliar por rodada, valores, ordem das
  amostras, coeficiente FedAvg e passos locais.
- Fazer cada variante auxiliar reconstruir localmente a mesma agenda
  determinística da comparação, sem compartilhar arquivos ou estado privado.
- Fazer o cliente adversário gerar localmente dados auxiliares sintéticos novos
  no início de cada rodada. Não reutilizar perfis nem valores entre amostras ou
  rodadas.
- Na referência, usar renovação por rodada não adaptativa: a política e a
  derivação de sementes são fixadas antes da execução, mas os dados são gerados
  dentro do cliente a cada rodada e não dependem das respostas do modelo global.
  Geração condicionada ao modelo pertence à ablação adaptativa.
- Durante seu treinamento, o adversário usa apenas nomes e valores auxiliares.
  Os nomes das vítimas são usados somente nas consultas de extração e nunca são
  associados a valores auxiliares nos exemplos de ataque.
- Permitir que o cliente adversário controle seus dados, função de perda,
  otimizador, passos locais e atualização submetida. Ele recebe somente o modelo
  global de cada rodada e nunca recebe atualizações individuais das vítimas.
- Usar como referência coeficiente FedAvg `1/11`, continuações positivas,
  treinamento somente na continuação e nenhuma transformação da atualização
  submetida. Tratar mistura positiva/negativa, outras capacidades,
  adaptatividade, escala do delta e coeficientes maiores como ablações.
- Na ablação prioritária de massa, manter um slot auxiliar físico e ponderá-lo
  como `k = 1..10` adversários virtuais: participação auxiliar `k/(10+k)` e
  participação de cada vítima `1/(10+k)`. Parear F0/F1 em cada `k`, parar na
  divisão 50/50, relatar todos os pontos e não confundir essa varredura com
  escala do delta submetido. Manter F2-F5 em `k=1` nesta versão.
- Tokenizar cada amostra de ataque uma vez. Mascarar os IDs exatos do prefixo e
  calcular a perda como média por conversa seguida da média do lote lógico.
- Aplicar DP-SGD ou substituição semântica somente aos 10 clientes-vítima.
- Usar a conversa que contém o registro completo como unidade de privacidade do
  DP-SGD. Não alegar DP no nível do participante enquanto as cinco conversas não
  forem agregadas como uma única unidade protegida.
- Recalcular e versionar os parâmetros do contabilizador de privacidade antes da
  campanha sempre que mudarem a unidade de privacidade, amostragem, lote, passos
  ou rodadas. Não reutilizar sigmas de uma configuração incompatível.
- Manter substituições estáveis por entidade e tipo e idênticas no par F4/F5.
  O nome permanece no conjunto transformado apenas porque é conhecimento prévio
  necessário ao gatilho; essa exceção não o reclassifica como dado não protegido.
- Manter os conjuntos das vítimas disjuntos. O adversário nunca pode acessar
  dados, valores protegidos, atualizações locais, arquivos do auditor ou mapas de
  substituição das vítimas.
- Manter o avaliador-oráculo separado de todos os clientes. O executor de
  consultas representa a capacidade de extração do mesmo ator adversário, mas
  não recebe o registro de respostas corretas.
- Manter o servidor honesto e limitado ao FedAvg simples. Ele não compartilha
  atualizações locais e não usa agregação segura nem robusta neste protocolo.
- Auditar o modelo inicial e o modelo global após cada rodada, com instruções e
  sementes de geração fixos, sem alterar o treinamento posterior.
- Usar, na rodada 20, a reprodução exata de pares corretos
  `nome conhecido -> tipo -> valor protegido` como métrica principal. Relatar
  também perfis completos, participantes com qualquer campo exposto, resultados
  por tipo, associações incorretas e exposições sem nome.
- Tratar a trajetória das rodadas como medidas descritivas repetidas, não como
  execuções independentes.
- Manter registros protegidos e mapas de substituição exclusivos do oráculo e
  fora do controle de versão. Relatar reproduções auxiliares apenas como
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
