import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# =============================
# 1. Importar os dados
# =============================
data_loaded = pd.read_pickle('x_scaled.pickle')
if isinstance(data_loaded, np.ndarray):
    df = pd.DataFrame(data_loaded)
else:
    df = data_loaded

# =============================
# 2. Análise Exploratória Inicial
# =============================

# 2.1 Estatísticas descritivas
print(df.describe().T)

# 2.2 Histograma de uma variável (exemplo: primeira)
plt.hist(df.iloc[:, 0], bins=30, edgecolor='black')
plt.title("Distribuição da Variável 0")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# 2.3 Matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Matriz de Correlação das Variáveis")
plt.show()

# =============================
# 3. Redução de Dimensionalidade (PCA)
# =============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
plt.title("Visualização dos Dados com PCA (2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()

# =============================
# 4. Determinação de eps com k-distance graph
# =============================
min_samples = 5  # Valor usado para exemplo. Ajustar conforme análise específica

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_pca)
distances, _ = neighbors_fit.kneighbors(X_pca)

# Ordenar distâncias do k-ésimo vizinho
distances = np.sort(distances[:, min_samples - 1])
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title(f'Gráfico k-distance (k = min_samples = {min_samples})')
plt.xlabel('Pontos ordenados')
plt.ylabel(f'Distância para o {min_samples}º vizinho mais próximo')
plt.grid(True)
plt.show()

# =============================
# 5. Aplicação do DBSCAN
# =============================
eps = 0.15  # Valor utilizado como exemplo. Escolher com base no gráfico k-distance.
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_pca)

# =============================
# 6. Visualização dos Clusters e Outliers
# =============================
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca,
    x='PC1', y='PC2',
    hue='Cluster',
    palette='tab10',
    style=(df_pca['Cluster'] == -1),
    markers={True: 'X', False: 'o'}
)
plt.title("Clusters e Outliers Detectados com DBSCAN")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.legend(title='Cluster')
plt.show()

# =============================
# 7. Análise dos Outliers
# =============================
outliers = df[labels == -1]
inliers = df[labels != -1]

# Comparar médias para identificar variáveis mais atípicas
outlier_means = outliers.mean()
inlier_means = inliers.mean()

diff = (outlier_means - inlier_means).abs().sort_values(ascending=False)
print("Variáveis com maiores diferenças entre outliers e dados normais:")
print(diff.head(10))
