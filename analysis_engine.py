from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

def descriptive_statistics(data):
    """Generate descriptive statistics of the dataset."""
    return data.describe()

def kmeans_clustering(data, n_clusters=3):
    """Apply K-Means clustering to the data."""
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_


def pca_analysis(data, n_components=2):
    """Apply PCA to reduce dimensionality of the data."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    return components
