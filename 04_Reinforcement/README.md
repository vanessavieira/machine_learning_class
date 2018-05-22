# Atividade 04 - Aprendizagem por Reforço

Para tudo que nos é enviado, assumimos que você está seguindo o código de honra a seguir.

## Código de Honra

>"Como membro da comunidade deste curso, não vou participar nem tolerar a desonestidade acadêmica".

## Objetivo da atividade
*Utilizar a aprendizagem por reforço para solução de um problema clássico.*

## Descrição da atividade
A atividade da equipe consiste em avaliar duas políticas diferentes para solução de um problema, nos moldes da aprendizagem por reforço (RL).

Nessa atividade vocês poderão utilizar um entre esses dois algoritmos de RL: Monte Carlo e Diferença Temporal.

## Descrição do problema: Monty Hall

O problema de Monty Hall, também conhecido por paradoxo de Monty Hall é um problema matemático e paradoxo que surgiu a partir de um concurso televisivo dos Estados Unidos chamado Let’s Make a Deal, exibido na década de 1970.

O jogo consistia no seguinte: Monty Hall, o apresentador, apresentava três portas aos concorrentes. Atrás de uma delas estava um prêmio (um carro) e as outras duas dois bodes.

- Na 1.ª etapa o concorrente escolhe uma das três portas (que ainda não é aberta);
- Na 2.ª etapa, Monty abre uma das outras duas portas que o concorrente não escolheu, revelando que o carro não se encontra nessa porta e revelando um dos bodes;
- Na 3.ª etapa Monty pergunta ao concorrente se quer decidir permanecer com a porta que escolheu no início do jogo ou se ele pretende mudar para a outra porta que ainda está fechada para então a abrir. Agora, com duas portas apenas para escolher — pois uma delas já se viu, na 2.ª etapa, que não tinha o prêmio — e sabendo que o carro está atrás de uma das restantes duas, o concorrente tem que tomar a decisão.

Qual é a estratégia mais lógica? Ficar com a porta escolhida inicialmente ou mudar de porta? Com qual das duas portas ainda fechadas o concorrente tem mais probabilidades de ganhar? Por quê?
Fonte: Wikipédia

Seu objetivo é criar as políticas de sempre mudar de porta e a de nunca mudar de porta e avaliar essas duas políticas.
