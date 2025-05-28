import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datetime import datetime
import warnings
import logging
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(series, period=20):
    """Calcula as Bandas de Bollinger."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def stochastic_oscillator(df, k_window=14, d_window=3):
    """Calcula o Oscilador Estocástico %K e %D."""
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    percent_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    percent_d = percent_k.rolling(window=d_window).mean()
    return percent_k, percent_d

def add_technical_indicators(df):
    """Adiciona indicadores técnicos ao DataFrame."""
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Upper_BB'], df['Lower_BB'] = bollinger_bands(df['Close'])
    df['Stochastic_%K'], df['Stochastic_%D'] = stochastic_oscillator(df)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    df.dropna(inplace=True)
    return df


class LSTMDataset(Dataset):
    def __init__(self, features, targets, time_steps=10):
        self.X, self.y = self.create_sequences(features, targets, time_steps)

    @staticmethod
    def create_sequences(features, targets, time_steps):
        Xs, ys = [], []
        for i in range(len(features) - time_steps):
            Xs.append(features[i:(i + time_steps)])
            ys.append(targets[i + time_steps])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def mape(y_true, y_pred):
    mape_vals = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != 0
        if np.any(mask):
            mape_i = np.mean(np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i])) * 100
        else:
            mape_i = np.nan
        mape_vals.append(mape_i)
    return np.array(mape_vals)

def preparar_dados():
    """Faz download e pré-processa os dados."""
    today = '2025-05-22'
    df = yf.download("^BVSP", start="2015-01-01", end=today, auto_adjust=True)
    df = add_technical_indicators(df)

    targets = df[['Close', 'High', 'Low']].shift(-1)
    df.dropna(inplace=True)
    features = df.drop(['Close', 'High', 'Low'], axis=1).values
    targets = targets.loc[df.index].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False)

    logging.info(f"Dados preparados: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, df

def treinar_lstm(X_train, y_train, time_steps=10, epochs=50, batch_size=32, device=None):
    """Treina o modelo LSTM."""
    train_ds = LSTMDataset(X_train, y_train, time_steps)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    input_size = X_train.shape[1]
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.2)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info(f"Treinando LSTM por {epochs} epochs no device {device}")
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.6f}")

    return model, device

def prever_lstm(model, X, y_scaler, time_steps=10, batch_size=32, device=None):
    """Gera previsões do modelo LSTM."""
    test_ds = LSTMDataset(X, np.zeros((X.shape[0],3)), time_steps)  # y dummy pois não usado
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).cpu().numpy()
            preds_scaled.append(out)
    preds_scaled = np.vstack(preds_scaled)
    preds = y_scaler.inverse_transform(preds_scaled)
    return preds

def treinar_xgb(X_train, y_train, X_test):
    """Treina o modelo XGBoost MultiOutput e gera previsões."""
    xgb_base = XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42,
    verbosity=0
)

    xgb = MultiOutputRegressor(xgb_base)
    xgb.fit(X_train, y_train)
    preds_scaled = xgb.predict(X_test)
    return xgb, preds_scaled

def avaliar_modelos(y_true, preds_dict):
    """Avalia múltiplos modelos imprimindo MAPE por variável."""
    logging.info("MAPE por modelo e variável:")
    for nome, pred in preds_dict.items():
        erro = mape(y_true, pred)
        logging.info(f"{nome:8s} | Close: {erro[0]:6.2f}% | High: {erro[1]:6.2f}% | Low: {erro[2]:6.2f}%")


def plotar_resultados(dates, y_true, preds_dict):
    # descobrir quantos pontos cada array tem
    lengths = [len(dates), len(y_true)] + [len(v) for v in preds_dict.values()]
    n = min(lengths)

    # cortar todas as séries para n pontos
    dates_cut = dates[-n:]
    y_cut = y_true[-n:]
    preds_cut = {name: preds[-n:] for name, preds in preds_dict.items()}

    # plotar todas as séries no mesmo gráfico
    plt.figure(figsize=(12, 6))

    # séries reais
    for i, label in enumerate(['Close', 'High', 'Low']):
        plt.plot(dates_cut, y_cut[:, i], label=f'Real - {label}', linewidth=2)

    # previsões
    for model_name, preds in preds_cut.items():
        for i, label in enumerate(['Close', 'High', 'Low']):
            plt.plot(dates_cut, preds[:, i], label=f'{model_name} - {label}', linestyle='--')

    plt.title("Previsões vs. Valores Reais – Conjunto de Teste")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Preparação dos dados
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, df = preparar_dados()

    # Treinamento LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_steps = 20
    # Linha no main():
    lstm_model, device = treinar_lstm(X_train, y_train, time_steps=time_steps, epochs=100, device=device)


    # Previsões LSTM no teste
    lstm_preds = prever_lstm(lstm_model, X_test, scaler_y, time_steps=time_steps, device=device)

    # Ajuste para alinhar verdadeiros e previsões LSTM (descontando time_steps)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_test_aligned = y_test_inv[time_steps:]

    # Treinamento XGBoost e previsão
    xgb_model, xgb_preds_scaled = treinar_xgb(X_train, y_train, X_test)
    xgb_preds = scaler_y.inverse_transform(xgb_preds_scaled)
    n_seq = min(len(lstm_preds), len(y_test_aligned))
    xgb_preds_aligned = xgb_preds[:n_seq]
    lstm_preds_aligned = lstm_preds[:n_seq]
    y_aligned = y_test_aligned[:n_seq]
    meta_inputs = np.hstack([lstm_preds_aligned, xgb_preds_aligned])

    mask = ~np.isnan(y_aligned).any(axis=1) & ~np.isnan(meta_inputs).any(axis=1)
    meta_inputs_clean = meta_inputs[mask]
    y_aligned_clean = y_aligned[mask]

    meta = LinearRegression()
    meta.fit(meta_inputs_clean, y_aligned_clean)
    ensemble_preds = meta.predict(meta_inputs_clean)

    # Avaliação
    preds_dict = {
        "LSTM": lstm_preds_aligned[mask],
        "XGBoost": xgb_preds_aligned[mask],
        "Ensemble": ensemble_preds
    }
    avaliar_modelos(y_aligned[mask], preds_dict)

    # Previsão para o próximo dia
    # Usar os últimos 'time_steps' registros de X_test para montar a sequência LSTM
    last_window = X_test[-time_steps:]  # shape (time_steps, n_features)

    # Previsão LSTM
    with torch.no_grad():
        lstm_input = torch.tensor(last_window.reshape(1, time_steps, -1), dtype=torch.float32).to(device)
        next_lstm_scaled = lstm_model(lstm_input).cpu().numpy()
    next_lstm = scaler_y.inverse_transform(next_lstm_scaled)

    # Previsão XGBoost (para o próximo dia)
    last_features = X_test[-1].reshape(1, -1)
    next_xgb_scaled = xgb_model.predict(last_features)
    next_xgb = scaler_y.inverse_transform(next_xgb_scaled)

    # Combinação ensemble
    next_meta_input = np.hstack([next_lstm, next_xgb])
    next_pred = meta.predict(next_meta_input).flatten()

    last_date = df.index[-1]

    logging.info(f"(Close, High, Low): {next_pred}")

    # Plotagem dos resultados
    dates = df.index[-n_seq:][mask]
    plotar_resultados(dates, y_aligned[mask], preds_dict)




if __name__ == "__main__":
    main()
