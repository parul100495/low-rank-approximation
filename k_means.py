import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as net
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = 7,7

def plot_adj_matrix(graph):
    adj_matrix = net.to_numpy_matrix(graph)
    plt.title("Plot of Adjacency Matrix")
    plt.imshow(adj_matrix, cmap = 'coolwarm')

def plot_spectral_embed(graph):
    plt.title("Plot of Spectral Embedding")
    net.draw_spectral(graph)

def plot_kmeans(graph):
    # Find adjacency matrix
    adj_matrix = net.adjacency_matrix(graph)
    
    # Eigen Decomposition
    eigen_values, eigen_vectors = np.linalg.eig(adj_matrix.toarray())
    # Getting the top 3 eigen values and eigen vectors that correspond
    # to the maximum values
    top3_indices = eigen_values.argsort()[::-1][:3]
    top3_vals = eigen_values[top3_indices]
    vec1, vec2, vec3 = eigen_vectors[:, top3_indices].T
    
    # For K-Means clustering, we need only the spectral embedding 
    # i.e; vec2 and vec3 (second and third vectors)
    spec = []
    for i in range(900):
        spec.append((vec2[i], vec3[i]))
        model = KMeans(n_clusters = 3).fit(spec)

    # Plot the points
    plt.scatter(
    vec2, vec3,
    s=50, c='lightblue',
    edgecolor='black',
    label='cluster points')

    # Plot the centroids
    plt.scatter(
    model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
    s=350, marker='*',
    c='purple', edgecolor='black',
    label='centroids')
    plt.legend(scatterpoints=1)
    plt.title('Result of KMeans Clustering')
    plt.xlabel('Second Eigenvector')
    plt.ylabel('Third Eigenvector')
    plt.grid()
    plt.show()

graph = net.planted_partition_graph(3, 300, 0.1, 0.02)

plot_adj_matrix(graph)

plot_spectral_embed(graph)

plot_kmeans(graph)
