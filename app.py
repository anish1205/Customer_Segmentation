# streamlit_kmeans_app.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load default dataset from URL

DEFAULT_URL = "https://raw.githubusercontent.com/sharmaroshan/Mall-Customer-Segmentation-Dataset/master/Mall_Customers.csv"


st.title("K-Means Clustering with PCA (3D)")
st.write("Upload your CSV or use the default Mall Customers dataset.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_URL)

st.write("### Raw Data", df.head())

# Select features
features = st.multiselect("Select features for clustering", options=df.columns, default=df.select_dtypes(include='number').columns.tolist())

if len(features) >= 2:
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)

    # Apply K-Means
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(df[features])
    df['Cluster'] = clusters

    # PCA for 3D
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(df[features])
    df[['PC1', 'PC2', 'PC3']] = pca_components

    st.write("### Clustered Data", df.head())

    # Plot 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA of Clusters')
    st.pyplot(fig)

else:
    st.warning("Please select at least 2 features for clustering.")
