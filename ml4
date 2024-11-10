import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("C:/Users/divya/Downloads/sales_data_sample.csv",encoding='ISO-8859-1')
data.head()
data.isnull().sum()
data.dropna(inplace=True)
features = data[['QUANTITYORDERED', 'SALES']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
for k in range(1, 11):  # Testing k from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()
optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters=kmeans.fit_predict(scaled_features)
print(data.head())
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('K-Means Clustering Results')
plt.xlabel('QUANTITYORDERED')
plt.ylabel('SALES')
plt.colorbar(label='Cluster')
plt.show()
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(scaled_features, method='ward'))
plt.title('Dendrogram')
plt.xlabel('QUANTITYORDERED')
plt.ylabel('SALES')
plt.show()
hc = AgglomerativeClustering(n_clusters=optimal_k)
hc_clusters = hc.fit_predict(scaled_features)
data['HC_Cluster'] = hc_clusters
