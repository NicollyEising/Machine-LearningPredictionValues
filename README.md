# README.md
## Descrição
Este projeto implementa um sistema de previsão diária do índice Bovespa (^BVSP) utilizando três abordagens de Machine Learning:
  - Rede LSTM (Long Short-Term Memory)
  - XGBoost (Extreme Gradient Boosting)
  - Ensemble com RandomForest em meta-features

O pipeline completo engloba:
  1. Download de dados históricos via `yfinance`
  2. Cálculo de indicadores técnicos (RSI, MACD, Bandas de Bollinger, ATR, ROC, etc.)
  3. Pré-processamento (normalização, criação de lags, codificação de data)
  4. Treinamento e validação de modelos LSTM e XGBoost
  5. Construção de meta-features e treino de ensemble
  6. Avaliação via MAPE (erro percentual absoluto médio)
  7. Geração de previsão para o próximo dia e plotagem de resultados

## Requisitos
  - Python ≥ 3.8
  - numpy
  - pandas
  - yfinance
  - matplotlib
  - scikit-learn
  - xgboost
  - torch (PyTorch)

## Como usar
  1. Ajustar parâmetros de download de dados no arquivo principal (`start`, `end`, ticker)
  2. Executar o arquivo main.py
  3. Serão exibidos logs de treinamento e plot de previsões versus valores reais
  4. A previsão para o próximo dia é impressa no console


## Organização do Código
  - `add_technical_indicators(df)` — adiciona indicadores e lags  
  - `LSTMDataset` e `LSTMModel` — definição do dataset e da rede  
  - `train_lstm(...)` / `predict_lstm(...)` — treino e predição LSTM  
  - `train_xgb(...)` — treino e predição XGBoost  
  - Pipeline de ensemble com `RandomForestRegressor`

## Métricas de Avaliação
  - MAPE para Close, High e Low  
  - Early stopping no treino da LSTM  
  - Logs detalhados de losses de treino e validação

