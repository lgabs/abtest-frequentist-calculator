# A/B Test Utilities

## Introdução
Classes e métodos para rodar testes A/B. Por enquanto, somente a análise Frequentista é suportada. A classe que lida com isso está no arquivo `abtests/frequentist_experiment.py`. O arquivo `abtests/frequentist_experiment_v2.py` é um arquivo temporário com alguns detalhes a mais para rodar várias análises duas a duas, nos casos de testes multivariados.

## Ambiente
Para testar localmente, crie um ambiente virtual. Verifique se você já tem o módulo `wheel` instalado e, depois, instale as dependências. Use os comandos do Makefile `make build` e `make install` para instalá-lo no ambiente. Caso queira testar as funções importando-as diretamente, adicione no seu path no script/notebook que for usar.

## Testes
Rode `pytest` para rodar os testes.