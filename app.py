
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, send_file
import pandas as pd
from data_processing import load_data, preprocess_data
from analysis_engine import (descriptive_statistics, kmeans_clustering, pca_analysis)
from report_generation import (visualize_descriptive_statistics,visualize_kmeans_clustering,visualize_pca,generate_report)
from sklearn.decomposition import PCA
import os

app = Flask(__name__)

data = None  # Global variable to store data
analysis_results = None  # Global variable to store analysis results

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and preprocessing."""
    global data
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = load_data(file_path)
        data = preprocess_data(data)
        return "File uploaded and data preprocessed successfully!"
    else:
        return "No file uploaded."

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global data, analysis_results
    
    # Ensure data is loaded
    if data is None:
        return render_template('index.html', message="No data available. Please upload a dataset first.")
    
    analysis_type = request.form.get('analysis')
    results = {}
    visualizations = []

    if analysis_type == 'kmeans':
        n_clusters = int(request.form.get('clusters', 3))
        labels = kmeans_clustering(data, n_clusters=n_clusters)
        results['KMeans'] = labels
        visualizations.append(visualize_kmeans_clustering(data, labels))
    
    
    elif analysis_type == 'pca':
        n_components = int(request.form.get('components', 2))
        components = pca_analysis(data, n_components)
        pca = PCA(n_components=n_components)
        explained_variance_ratio = pca.fit(data).explained_variance_ratio_
        results['PCA'] = components
        visualizations.append(visualize_pca(components, explained_variance_ratio))
    
    elif analysis_type == 'statistics':
        results['Descriptive Statistics'] = descriptive_statistics(data)
        visualizations += visualize_descriptive_statistics(data)
    
    analysis_results = results
    return render_template('index.html', columns=data.columns.tolist(), message="Analysis completed!", results=results, visualizations=visualizations)


@app.route('/report', methods=['GET'])
def generate_report_page():
    """Generate a report and offer it for download."""
    global data, analysis_results
    if data is not None and analysis_results is not None:
        report_path = 'report.txt'
        generate_report(data, analysis_results)
        return send_file(report_path, as_attachment=True)
    else:
        return "No data or analysis results found."

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create upload folder if it doesn't exist
    app.run(debug=True)
