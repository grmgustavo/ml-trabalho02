import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as pltz
import os
import matplotlib.pyplot as plt

# Definir o caminho para o diretório de dados
caminho_dados = "./dados/"

# Carregar os dados (dolar, bvsp, petroleo, ouro, gol) e definir o índice como a coluna 'Date' (para o dolar e o bvsp) ou 'Data' (para o petroleo, ouro e gol) 
# e converter as datas para o formato datetime (parse_dates=True)
dolar = pd.read_csv(os.path.join(caminho_dados, 'BRL=X.csv'), index_col='Date', parse_dates=True)
bvsp = pd.read_csv(os.path.join(caminho_dados, '^BVSP.csv'), index_col='Date', parse_dates=True)
petroleo = pd.read_csv(os.path.join(caminho_dados, 'Petroleo Tratado.csv'), parse_dates=['Data'], dayfirst=True, date_format='%d%m%Y')
ouro = pd.read_csv(os.path.join(caminho_dados, 'Ouro Tratado.csv'), parse_dates=['Data'], dayfirst=True, date_format='%d%m%Y')
gol = pd.read_csv(os.path.join(caminho_dados, 'GOLL4.SA.csv'), index_col='Date', parse_dates=True)

# Ordenar os DataFrames por data
dolar = dolar.sort_index()
bvsp = bvsp.sort_index()
petroleo = petroleo.sort_values('Data').reset_index(drop=True)
ouro = ouro.sort_values('Data').reset_index(drop=True)
gol = gol.sort_index()

# Juntar os DataFrames em um único DataFrame (df)
df = pd.concat([dolar['Close'], bvsp['Close'], petroleo['Ultimo'], ouro['Ultimo'], gol['Close']], axis=1)
# Renomear as colunas para facilitar a identificação das características e do alvo
df.columns = ['Dolar', 'BVSP', 'Petroleo', 'Ouro', 'GOL']

# Verificar e tratar dados ausentes se necessário
df = df.fillna(df.mean())

# Definir características (X) e alvo (y)
X = df[['Dolar', 'Petroleo', 'Ouro', 'BVSP']]
y = df['GOL']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escolher e treinar um modelo de regressão (por exemplo, Linear Regression)
model = LinearRegression()
# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar o desempenho
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

# Imprimir os resultados das métricas de avaliação de desempenho (mse, mae, rmse)
print(f'Mean Squared Error(Erro Médio Quadrático): {mse}') # em pt = Erro Médio Quadrático
print(f'Mean Absolute Error(Erro Médio Absoluto): {mae}') # em pt = Erro Médio Absoluto
print(f'Root Mean Squared Error(Raiz do Erro Médio Quadrático): {rmse}') # em pt = Raiz do Erro Médio Quadrático

# Plotar os resultados (preço real vs previsões)
plt.figure(figsize=(15, 5))
plt.plot(y_test.index.astype(str), y_test.values, label='Preço Real')
plt.plot(y_test.index.astype(str), predictions, label='Previsões')
plt.legend()
plt.savefig('resultado_plot.png') # Salvar o gráfico como imagem ou plt.show() para exibir o gráfico na tela, caso esteja executando o código em um notebook