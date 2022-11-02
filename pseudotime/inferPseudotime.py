####DATA: LINHAS = CELULAS // COLUNAS == GENES

import torch
import numpy as np
from matplotlib import pyplot as plt
from slingshot import Slingshot
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import random as r
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from time import perf_counter


targets = pd.read_hdf('../data/train_cite_targets.h5', start=0, index_col=0)

print(targets)

ED = targets.T

#Redução de dimensionalidade para inferência do pseudotime.
#TSNE não é recomendado para 50+ características, entretanto Pratapa utiliza TSNE para visualizar as trajetórias
#TSNE para o train_cite_inputs.h5 rodou por mais de 22h sem concluir
#KernelPCA, kernel='poly' foi executado completamente em pouco mais de 1h (não testei outros kernels e acredito ter impacto não só no tempo mas também nos resultados)

p = 400
t1_start = perf_counter()
print("inicio tsne")
#tsne = TSNE(n_components=2,perplexity=p).fit_transform(ED.T.values)
#tsne = PCA(n_components=2, svd_solver='full').fit_transform(ED.T.values)
tsne = KernelPCA(n_components=2, kernel='poly').fit_transform(ED.T.values) ##kernel era linear
#tsne = TSNE(n_components=2,perplexity=p).fit_transform(tsne2)
print("fim tsne")
t1_stop = perf_counter()
print("Tempo da redução de dimensionalidade:", t1_stop-t1_start)

tdf = pd.DataFrame(tsne, columns=['t-SNE 1', 't-SNE 2'],index=ED.columns)

cell_names = tdf.T.columns
print("Numero de celulas: ", len(cell_names))


format_data = []
for cell_name in cell_names:
    format_data.append(list(tdf.T[cell_name]))
data = np.array(format_data)                     

    
print("Calculando kmeans")

## Cálculo de k-means para determinar tipos celulares (os labels podem ser diretamente os tipos celulares já anotados)

kmeans_model2 = KMeans(n_clusters=3, random_state=10).fit(data) ##min clusters depende dos dados, pq?
cluster_labels = np.array(kmeans_model2.labels_)

print("Fim calculo kmeans")



## Determina a quantidade de genes sem expressão e calcula a variação da expressão dos genes
n_zeros = 0

dict_var = {}

for gene in ED.T:
    v_local = []
    v_local.append(0)
    v_local.append(np.var(ED.T[gene]))
    if np.var(ED.T[gene]) == 0:
        n_zeros += 1
    dict_var[gene] = v_local

print("TOTAL ZEROS: ", n_zeros)

currentIndexPD = ['VGAMpValue','Variance']
outFileName = 'GeneOrdering.csv'
df = pd.DataFrame(data=dict_var, index=currentIndexPD)
df.T.to_csv(outFileName)    


#plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)

cluster_labels_onehot = np.zeros((cluster_labels.shape[0], cluster_labels.max()+1))
cluster_labels_onehot[np.arange(cluster_labels.shape[0]), cluster_labels] = 1

print(data.shape)
#plt.show()


# Inferência do(s) pseudotime(s)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)

slingshot = Slingshot(data, cluster_labels_onehot, debug_level='verbose')

slingshot.fit(num_epochs=10, debug_axes=axes)

plt.savefig("slingshot1.png");

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].set_title('Clusters')
axes[1].set_title('Pseudotime')
slingshot.plotter.curves(axes[0], slingshot.curves)
slingshot.plotter.clusters(axes[0], labels=np.arange(slingshot.num_clusters), s=4, alpha=0.5)
slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)
plt.savefig("slingshot2.png");

# Exportação dos valores de pseudotime unificados inferidos

currentIndexPT = ['PseudoTime']
normalized_pt = [pt/max(slingshot.unified_pseudotime) for pt in slingshot.unified_pseudotime]

dict_PT = {}

for i in range(len(cell_names)):
    currentCell = cell_names[i]
    currentPT = normalized_pt[i]
    dict_PT[currentCell] = currentPT

outFileName = 'PseudoTime.csv'
dfPT = pd.DataFrame(data=dict_PT, index=currentIndexPT)
dfPT.T.to_csv(outFileName)    
