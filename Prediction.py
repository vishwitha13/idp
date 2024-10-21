import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
# Fetch dataset
heart_disease = fetch_ucirepo(id=45)

# Data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# Get column names from metadata
column_names = getattr(heart_disease.variables, 'feature_names', None)

# If feature_names is not available, you may need to explore the variables attribute
# column_names = heart_disease.variables  # Uncomment this line and explore the variables attribute

# Convert data to pandas DataFrame
df = pd.DataFrame(data=X, columns=column_names)

# Add the target column to the DataFrame
df['target'] = y

# Display the DataFrame

df['target'] = df['target'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

# Assuming you have already loaded your dataset into pandas DataFrame 'df'

# Separate features (X) and target (y)
X = df.drop(columns=['target'])  # Assuming 'target' is the column name for the labels
y = df['target']

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Convert y to a numpy array for indexing
y_np = y.to_numpy()

# Perform cross-validation
accuracies = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_np[train_index], y_np[test_index]

    # Impute missing values in the training and testing data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Initialize KMeans with optimized parameters
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=50, max_iter=500, random_state=42)

    # Fit KMeans on the scaled training data
    kmeans.fit(X_train_scaled)

    # Predict cluster labels for the testing data
    test_cluster_labels = kmeans.predict(X_test_scaled)

    # Assign labels based on majority class in each cluster
    cluster_0_label = np.bincount(y_test[test_cluster_labels == 0]).argmax()
    cluster_1_label = np.bincount(y_test[test_cluster_labels == 1]).argmax()
    predicted_labels = np.where(test_cluster_labels == 0, cluster_0_label, cluster_1_label)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    accuracies.append(accuracy)

# Average accuracy across folds
avg_accuracy = np.mean(accuracies)
print("Average Cross-Validation Accuracy with KMeans:", avg_accuracy)


filename = 'heart-disease-prediction-kmeans-model.pkl'
pickle.dump(kmeans, open(filename, 'wb'))