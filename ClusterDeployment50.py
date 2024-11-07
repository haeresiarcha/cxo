#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans  # Fallback for no categorical data


# Set up Streamlit app layout
st.title("Cluster Analysis of Cities")
st.sidebar.title("Cluster Settings")

# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv('cities_standardized.csv')
    return df

df = load_data()

# Separate features and target
X = df.drop(['City'], axis=1)

# Identify categorical and numeric columns
categorical_columns = ['Grouping', 'Capital Status']
numeric_columns = [col for col in X.columns if col not in categorical_columns]

# Sidebar for selecting clusters (k)
k_min, k_max = 2, 10
k_clusters = st.sidebar.slider('Select number of clusters (k)', k_min, k_max, 3)

### 1. Dynamic Data Filtering
st.sidebar.subheader("Filter Options")
selected_features = st.sidebar.multiselect("Select Numeric Features", numeric_columns, default=numeric_columns)
selected_capital_status = st.sidebar.multiselect("Select Capital Status", df['Capital Status'].unique(), default=df['Capital Status'].unique())
selected_grouping = st.sidebar.multiselect("Select Grouping", df['Grouping'].unique(), default=df['Grouping'].unique())

# Filter the dataset based on sidebar selections
filtered_df = df[(df['Capital Status'].isin(selected_capital_status)) & (df['Grouping'].isin(selected_grouping))]

# Check if any categorical columns are selected
selected_categorical_columns = [col for col in categorical_columns if col in selected_features]
selected_numeric_columns = [col for col in selected_features if col in numeric_columns]
X_filtered = filtered_df[selected_features]

# Standardize numeric columns
scaler = StandardScaler()
X_filtered_scaled = X_filtered.copy()
if selected_numeric_columns:
    X_filtered_scaled[selected_numeric_columns] = scaler.fit_transform(X_filtered[selected_numeric_columns])

# Check if we have categorical columns to handle
if selected_categorical_columns:
    # Convert categorical columns to strings for KPrototypes
    X_filtered_scaled[selected_categorical_columns] = X_filtered_scaled[selected_categorical_columns].astype(str)

    # Get indices of categorical columns
    categorical_indices = [X_filtered_scaled.columns.get_loc(col) for col in selected_categorical_columns]

    # Apply K-Prototypes clustering
    kproto = KPrototypes(n_clusters=k_clusters, random_state=42)
    clusters = kproto.fit_predict(X_filtered_scaled.values, categorical=categorical_indices)
else:
    # No categorical data, fallback to KMeans
    st.write(".")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_filtered_scaled.values)

# Add cluster labels to the original filtered dataframe
filtered_df['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
if selected_numeric_columns:
    X_pca = pca.fit_transform(X_filtered_scaled[selected_numeric_columns])
else:
    # If no numeric columns are selected, create dummy PCA components
    X_pca = np.zeros((len(X_filtered_scaled), 2))
filtered_df['PCA1'] = X_pca[:, 0]
filtered_df['PCA2'] = X_pca[:, 1]

# Define a mapping from numeric labels to letters
cluster_mapping = {i: chr(65 + i) for i in range(k_clusters)}
filtered_df['Cluster'] = filtered_df['Cluster'].map(cluster_mapping)

### 2. Isolate Cities by Cluster for PCA Plot
st.sidebar.subheader("PCA Plot Settings")
selected_cluster_for_plot = st.sidebar.multiselect(
    "Select Cluster(s) to Isolate in PCA Plot", options=filtered_df['Cluster'].unique(), default=filtered_df['Cluster'].unique()
)
filtered_plot_df = filtered_df[filtered_df['Cluster'].isin(selected_cluster_for_plot)]

# Plot the PCA results with city labels
st.subheader("PCA Plot of City Clusters")
palette = sns.color_palette('viridis', k_clusters)
cluster_colors = {label: palette[i % len(palette)] for i, label in enumerate(sorted(cluster_mapping.values()))}

fig, ax = plt.subplots(figsize=(16, 12))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster',
    palette=cluster_colors, data=filtered_plot_df, s=150,
    alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax
)
ax.set_title('City Clusters', fontsize=22)
ax.set_xlabel('PCA1', fontsize=18)
ax.set_ylabel('PCA2', fontsize=18)

# Annotate city names
for i, city in enumerate(filtered_plot_df['City']):
    ax.text(filtered_plot_df['PCA1'].iloc[i] + 0.03, filtered_plot_df['PCA2'].iloc[i] + 0.03, city, fontsize=10, alpha=0.75, weight='bold')

st.pyplot(fig)

### 3. Nearest Neighbors Feature
st.sidebar.subheader("Find Urban Peers")
selected_city = st.sidebar.selectbox("Select a City", filtered_df['City'].unique())

# Find the five closest neighbors
def find_closest_neighbors(city, df, numeric_columns):
    city_data = df[df['City'] == city][numeric_columns].values
    if len(city_data) == 0:
        return []
    city_data = city_data[0]
    cluster = df[df['City'] == city]['Cluster'].values[0]
    cluster_data = df[df['Cluster'] == cluster]

    distances = []
    for i, row in cluster_data.iterrows():
        if row['City'] != city:
            other_city_data = row[numeric_columns].values
            distance = euclidean(city_data, other_city_data)
            distances.append((row['City'], distance, row['PCA1'], row['PCA2']))

    # Sort by distance
    distances.sort(key=lambda x: x[1])
    return distances[:5]

# Get the closest neighbors for the selected city
closest_neighbors = find_closest_neighbors(selected_city, filtered_df, selected_numeric_columns)

# Display closest cities in a table
st.subheader(f"Five Closest Cities to {selected_city}")
if closest_neighbors:
    neighbor_df = pd.DataFrame(closest_neighbors, columns=['City', 'Distance', 'PCA1', 'PCA2'])
    st.write(neighbor_df[['City', 'Distance']])
else:
    st.write("Not enough data to find neighbors.")

# Plot the PCA results with city labels and neighbor connections
st.subheader(f"PCA Plot for {selected_city} and Its Peers")
fig, ax = plt.subplots(figsize=(16, 12))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster',
    palette=cluster_colors, data=filtered_df, s=150,
    alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax
)

# Highlight the selected city
city_pca = filtered_df[filtered_df['City'] == selected_city][['PCA1', 'PCA2']].values
if len(city_pca) > 0:
    city_pca = city_pca[0]
    plt.scatter(city_pca[0], city_pca[1], color='red', s=200, label=f'{selected_city}', edgecolor='black', zorder=5)

    # Draw connections and plot closest neighbors
    for city_name, dist, pca1, pca2 in closest_neighbors:
        plt.plot([city_pca[0], pca1], [city_pca[1], pca2], color='gray', linestyle='--', linewidth=0.7)
        plt.scatter(pca1, pca2, color='orange', s=100, edgecolor='black', zorder=5)

    ax.legend()
else:
    st.write(f"{selected_city} is not available in the filtered data.")

ax.set_title(f'PCA Plot Highlighting {selected_city} and Peers', fontsize=22)
ax.set_xlabel('PCA1', fontsize=18)
ax.set_ylabel('PCA2', fontsize=18)

st.pyplot(fig)

### 4. Cluster Comparison Table
st.subheader("Cluster Comparison Table")
cluster_summary = filtered_df.groupby('Cluster')[selected_numeric_columns].mean().reset_index()
st.write(cluster_summary)

### 5. Cluster Profile Viewer
st.sidebar.subheader("Cluster Profile Viewer")
selected_cluster = st.sidebar.selectbox("Select Cluster to view", filtered_df['Cluster'].unique())

# Show profile of the selected cluster
st.subheader(f"Profile of Cluster {selected_cluster}")
st.write(filtered_df[filtered_df['Cluster'] == selected_cluster].describe())

### 6. Radar Chart for Cluster Characteristics
st.subheader("Radar Chart of Cluster Characteristics")
def radar_chart(df, cluster_col='Cluster', numeric_columns=None):
    if numeric_columns is None or len(numeric_columns) == 0:
        st.write("No numeric columns selected for radar chart.")
        return

    cluster_means = df.groupby(cluster_col)[numeric_columns].mean()
    num_vars = len(numeric_columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i, (cluster, row) in enumerate(cluster_means.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {cluster}', color=palette[i % len(palette)])
        ax.fill(angles, values, color=palette[i % len(palette)], alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_columns)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    st.pyplot(fig)

radar_chart(filtered_df, cluster_col='Cluster', numeric_columns=selected_numeric_columns)

### 7. Boxplot of Numeric Features by Cluster
st.subheader("Boxplots of Numeric Features by Cluster")
if selected_numeric_columns:
    fig, axes = plt.subplots(len(selected_numeric_columns), figsize=(15, 5 * len(selected_numeric_columns)))
    if len(selected_numeric_columns) == 1:
        axes = [axes]
    for i, col in enumerate(selected_numeric_columns):
        sns.boxplot(x='Cluster', y=col, data=filtered_df, palette=palette, ax=axes[i])
        sns.swarmplot(x='Cluster', y=col, data=filtered_df, color=".25", ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No numeric columns selected for boxplots.")

