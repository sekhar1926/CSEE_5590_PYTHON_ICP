import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans


# reading collge dataset
dataset = pd.read_csv('College.csv')
data_frame = dataset.iloc[:,[9,10]]

#normalizing and preprocessing Data
x=((data_frame-data_frame.min())/(data_frame.max()-data_frame.min()))*20
print(x);
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


X_scaled.sample(5)
nclusters = 3# this is K
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)


#computing silhouette_score
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print('silhouette_score :', score)


#Plotting Clusters
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2: 'g',
                   3:'y',
                   4:'b'
                   }
label_color = [LABEL_COLOR_MAP[l] for l in km.predict(X_scaled)]
plt.scatter(X_scaled_array[:, 0], X_scaled_array[:, 1], c=label_color)
plt.title("clustering Based on Outstate and Room.Board data in college.csv")
plt.show()

