from ipaddress import collapse_addresses
import os
from this import d
import numpy as np
import scipy
import pandas as pd
import tables
import matplotlib.pyplot as plt

import umap
import umap.plot 

import hdbscan
from slingshot import Slingshot


DATA_DIR             = "/home/camata/git/cgp-grn/data/"
FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CELL_METADATA     = os.path.join(DATA_DIR,"metadata.csv")

cite_train = pd.read_hdf(FP_CITE_TRAIN_INPUTS,start=65000, index_col=0)

cell_names = []
drop_genes = []
dict_var = {}
for gene in cite_train.columns:
    variance  = np.var(cite_train[gene])
    if variance==0:
        drop_genes.append(gene)
    else:
        dict_var[gene] = variance

dfv = pd.DataFrame(data=dict_var, index=['variance'])
dfv.T.to_hdf('genes_variance.h5',key='variance');    
del dfv 
        
print("Number of genes with no variances", len(drop_genes))

# remove genes with no variances
dropped = cite_train.drop(columns=drop_genes)
cell_names = dropped.T.columns
print("Number of cells: %d" %(dropped.shape[0]))
print("Number of genes: %d" %(dropped.shape[1]))

# dimension reduction using UMAP
print("Running umap...")
mapper = umap.UMAP( n_neighbors=50,min_dist=0.0,n_components=2,n_epochs=30,low_memory=True).fit_transform(dropped.values)

plt.scatter(mapper[:, 0], mapper[:, 1],s=0.1)
plt.savefig('umap_scatter.png')

# cluster
print("Running hdbscan...")
hdbscan_labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=500,
).fit_predict(mapper)

clustered = (hdbscan_labels >= 0)
labels = hdbscan_labels

plt.scatter(mapper[~clustered, 0],
            mapper[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(mapper[clustered, 0],
            mapper[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')
plt.savefig("cluster_output.png")

print("Number of clusters: %d" %(labels.max()+1))

cluster_labels_onehot = np.zeros((labels.shape[0], labels.max()+1))
cluster_labels_onehot[np.arange(labels.shape[0]), labels] = 1

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)

print("Running Slingshot...")
slingshot = Slingshot(mapper, cluster_labels_onehot, debug_level='verbose')
slingshot.fit(num_epochs=10, debug_axes=axes)

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
axes[0].set_title('Clusters')
axes[1].set_title('Pseudotime')
slingshot.plotter.curves(axes[0], slingshot.curves)
slingshot.plotter.clusters(axes[1], color_mode='pseudotime', s=5)

plt.savefig("slingshot_output.png");
print("pseudotime size %d" % len(slingshot.unified_pseudotime) )


max_ptime = max(slingshot.unified_pseudotime)
normalized_pt = [pt/max_ptime for pt in slingshot.unified_pseudotime]
dict_PT = {}

for i in range(len(cell_names)):
    currentCell = cell_names[i]
    currentPT   = normalized_pt[i]
    dict_PT[currentCell] = currentPT

outFileName = 'cell_pseudotime.h5'
dfPT = pd.DataFrame(data=dict_PT, index=['pseudotime'])
dfPT.T.to_hdf(outFileName,key='pseudotime')
