import os
from matplotlib import pyplot as plt
import pandas as pd
from analysis_engine import  (descriptive_statistics, kmeans_clustering, pca_analysis)
import seaborn as sns
from sklearn.decomposition import PCA

filename=r'C:\Users\91766\my_flask_app\static\save'
def save_plot(filename):
    """Save the plot to the save folder inside the static directory."""
    static_save_folder = os.path.join('static', 'save')
    os.makedirs(static_save_folder, exist_ok=True)  # Ensure folder exists
    filepath = os.path.join(static_save_folder, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath 

def visualize_descriptive_statistics(data):
    """Visualize the descriptive statistics using box plots and histograms."""
    plt.figure(figsize=(12, 8))
    
    # Box plot for all numeric features
    sns.boxplot(data=data)
    plt.title('Box Plot of Numerical Features')
    box_plot_path = save_plot('box_plot.png')

    # Histograms for each feature
    histogram_paths = []
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Distribution of {column}')
        histogram_path = save_plot(f'histogram_{column}.png')
        histogram_paths.append(histogram_path)

    return [box_plot_path] + histogram_paths


from sklearn.decomposition import PCA

def visualize_kmeans_clustering(data, kmeans_labels):
    """Visualize the K-Means clustering results."""
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    plt.title('K-Means Clustering')
    plt.colorbar()
    

    return save_plot('kmeans_clustering.png')

def visualize_pca(components, explained_variance_ratio):
    """Visualize the PCA results (2D scatter plot)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(components[:, 0], components[:, 1], cmap='plasma')
    plt.title('PCA: First Two Principal Components')
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2f} variance)')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2f} variance)')
    plt.grid(True)
    return save_plot('pca_scatter.png')

def generate_report(data, analysis_results):
    """Generate a text report and visualizations based on analysis results."""
    with open('report.txt', 'w') as file:
        file.write("Data Analysis Report\n")
        file.write("====================\n")
        file.write("Summary Statistics:\n")
        file.write(str(descriptive_statistics(data)))
        file.write("\n\nAnalysis Results:\n")
        file.write(str(analysis_results))
    
    # Generate and save visualizations
    visualizations = []
    if 'KMeans' in analysis_results:
        img_path = visualize_kmeans_clustering(data, analysis_results['KMeans'])
        visualizations.append(img_path)

    if 'PCA' in analysis_results:
        components = analysis_results['PCA']
        explained_variance_ratio = PCA(n_components=2).fit(data).explained_variance_ratio_
        img_path = visualize_pca(components, explained_variance_ratio)
        visualizations.append(img_path)

    if 'Descriptive Statistics' in analysis_results:
        img_paths = visualize_descriptive_statistics(data)
        visualizations.extend(img_paths)

    return visualizations
