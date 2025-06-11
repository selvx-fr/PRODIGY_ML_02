# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Dataset
df = pd.read_csv("Mall_Customers.csv")  # Ensure the CSV file is in your project folder

# Step 3: Select Features for Clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Step 5: Use the Elbow Method to Determine Optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Step 6: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

# Step 7: Apply KMeans with Optimal Clusters (k = 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 8: Visualize Clusters in 2D
plt.figure(figsize=(18, 5))

# Age vs Spending Score
plt.subplot(1, 3, 1)
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
plt.title('Clusters: Age vs Spending Score')

# Income vs Spending Score
plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
plt.title('Clusters: Income vs Spending Score')

# Age vs Income
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Cluster', palette='Set2', s=100)
plt.title('Clusters: Age vs Income')

plt.tight_layout()
plt.show()

# Step 9: Save Clustered Data to CSV
df.to_csv("Clustered_Customers.csv", index=False)
print("Clustering complete. Results saved to 'Clustered_Customers.csv'")