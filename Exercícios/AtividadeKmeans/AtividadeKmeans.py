import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Importar o dataset
seeds_df = pd.read_csv('seeds.csv', header=None)

# Converter para array numpy
seeds_array = seeds_df.to_numpy()

# Separar em treino (70%) e teste (30%)
X_train, X_test = train_test_split(seeds_array, test_size=0.3, random_state=42)

# APLICANDO ALGORITMO
# Instanciar modelo com 3 clusters (pois há 3 tipos de sementes no dataset)
model = KMeans(n_clusters=3, random_state=42)

# Treinar o modelo com o conjunto de treino
model.fit(X_train)

# Obter os rótulos (clusters) do conjunto de teste
labels = model.predict(X_test)

# Visualizando os clusters
# Selecionar duas features para plot (por exemplo, colunas 0 e 1)
x = X_test[:, 0]
y = X_test[:, 1]

# Criar o gráfico de dispersão
plt.figure(facecolor='lightblue')
plt.scatter(x, y, c=labels, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters gerados pelo K-Means")
plt.show()