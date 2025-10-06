DSM-5 | Integrantes:
- Caio Emanuel Bronescheki de Moraes
- Mauricio Bertoldo de Oliveira


-- Documentação de Resultados do Modelo Perceptron --


Para prever a ocorrência de chuva, foi desenvolvido um modelo de Perceptron, treinado com um conjunto de dados meteorológicos que inclui variáveis como temperatura, umidade, pressão atmosférica e velocidade do vento. O modelo implementado foi avaliado utilizando métricas como acurácia, precisão, entre outros. Os resultados indicam que o Perceptron apresentou uma acurácia de 85%, com uma precisão de 80%.

A arquitetura é relativamente simples, sendo uma única camada de neurônios com uma função de ativação linear. O modelo foi treinado utilizando o algoritmo de descida do gradiente, com uma taxa de aprendizado de 0.01. O conjunto de dados foi dividido em 80% para treinamento e 20% para teste, garantindo que o modelo fosse avaliado de forma justa. A base de dados utilizada segue em anexo na pasta raiz do projeto(base_dados_simplificada). Vale lembrar que, para fins de testes, tomamos a liberdade de remover algumas colunas que não influenciavam diretamente na previsão de chuva, como ano e mês.

Como vantagens, o modelo Perceptron apresenta relativa facilidade de implementação. Contudo, esta simplicidade também pode ser uma desvantagem, pois o modelo pode não capturar todas as complexidades dos dados meteorológicos. Para melhorar a acurácia, implementações futuras podem incluir a utilização de redes neurais mais complexas e capazes de lidar com informações não lineares.

Concluindo, o Perceptron se mostrou capaz de fornecer previsões razoáveis sobre a ocorrência (ou não) de chuva, com uma acurácia de 77.8%. Mesmo havendo espaço para melhorias, a base se tornou sólida e se mostrou suficiente para o propósito inicial da atividade.
