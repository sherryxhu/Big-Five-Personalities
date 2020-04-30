import pandas as pd
import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Dimensionality reduction
from sklearn.manifold import TSNE
data_file = "data-final.csv"
df = pd.read_csv(data_file, sep="\t")
df_q = df.iloc[:, :100]
df_q_clean = df_q.dropna()
from sklearn import preprocessing
normalized_vectors = preprocessing.normalize(df_q_clean)
scores = []
for i in range(20,33):
    res = KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_
    scores.append(res)
    print(i, res)
