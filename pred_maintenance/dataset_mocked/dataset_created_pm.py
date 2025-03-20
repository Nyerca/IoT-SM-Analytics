import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
import time

# Load dataset
df = pd.read_csv("../dataset/predictive_maintenance_data.csv")
df = df[df['label'] != 2] # Remove broken label

# Convert timestamp to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Drop machine_id and timestamp for ML model
df.drop(columns=["machine_id", "timestamp"], inplace=True)

# EDA: Summary statistics
print(df.describe())

# EDA: Distribution plots
plt.figure(figsize=(12, 6))
sns.histplot(df["temperature"], kde=True, bins=30, color='blue', label='Temperature')
sns.histplot(df["vibration"], kde=True, bins=30, color='green', label='Vibration')
sns.histplot(df["pressure"], kde=True, bins=30, color='red', label='Pressure')
plt.legend()
plt.title("Feature Distributions")
plt.show()

# EDA: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# Splitting dataset
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
log_clf = LogisticRegression(max_iter=500)
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier(n_estimators=100)
gb_clf = GradientBoostingClassifier(n_estimators=100)
mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500)
svm_clf = SVC(probability=True)

# Function to train and evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_train_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train_time

    start_predict_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_predict_time

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    total_time = train_time + predict_time

    return accuracy, recall, precision, f1, mcc, train_time, predict_time, total_time

# Store the results
results = []

# List of classifiers
classifiers = [log_clf, dt_clf, rf_clf, gb_clf, mlp_clf, svm_clf]
classifier_names = ['Logistical Classification', 'Decision Tree', 'Random Forest',
                    'Gradient Boosting Classifier', 'Neural Network MLP', 'SVM']

# Evaluate each classifier
for clf, name in zip(classifiers, classifier_names):
    accuracy, recall, precision, f1, mcc, train_time, predict_time, total_time = evaluate_model(clf, X_train, y_train, X_test, y_test)
    results.append([name, accuracy, recall, precision, f1, mcc, train_time, predict_time, total_time])

# Create a DataFrame with the results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "Precision", "F1-Score", "MCC score",
                                            "time to train", "time to predict", "total time"])

# Print the results in the desired format
pd.set_option('display.float_format', '{:,.6f}'.format)
print(results_df.to_string(index=False))






# Initialize the VotingClassifier using majority vote rule (hard voting)
voting_clf = VotingClassifier(estimators=[
    ('log_clf', log_clf),
    ('dt_clf', dt_clf),
    ('rf_clf', rf_clf),
    ('gb_clf', gb_clf),
    ('mlp_clf', mlp_clf),
    ('svm_clf', svm_clf)
], voting='hard')  # 'hard' voting for majority vote rule

# Evaluate the majority vote classifier
accuracy, recall, precision, f1, mcc, train_time, predict_time, total_time = evaluate_model(voting_clf, X_train, y_train, X_test, y_test)

# Store the result for the majority vote classifier
results.append(['Majority Vote Classifier', accuracy, recall, precision, f1, mcc, train_time, predict_time, total_time])

# Create a DataFrame with the updated results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "Precision", "F1-Score", "MCC score",
                                            "time to train", "time to predict", "total time"])

# Print the results including the majority vote classifier
pd.set_option('display.float_format', '{:,.6f}'.format)
print(results_df.to_string(index=False))









# New data sample as a string (you can replace this with a CSV file path if needed)
new_data = """
machine_id,timestamp,temperature,vibration,pressure,label
1,2025-01-01 00:58:00,39.62926582297012,5.5743091483273775,103.3688849960026,0
1,2025-01-01 06:08:00,39.752563949797675,5.5066683582684375,102.49600169657465,0
1,2025-02-06 10:03:00,43.146048476084715,7.064520567730961,116.68026028204056,1
1,2025-01-01 02:00:00,39.80141318116897,5.7318723115071695,102.74327293108695,0
"""

# Convert the new data into a DataFrame
from io import StringIO

# Read the data into a pandas DataFrame
new_data_df = pd.read_csv(StringIO(new_data))

# Drop unnecessary columns (machine_id, timestamp, label) before prediction
new_data_for_prediction = new_data_df.drop(columns=["machine_id", "timestamp", "label"])
# Scale the new data using the same scaler used for the training data
new_data_scaled = scaler.transform(new_data_for_prediction)
# Make predictions on the new data using the majority vote classifier
new_predictions = voting_clf.predict(new_data_scaled)

# Print the predictions
print("\nPredictions for the new data:")
print(new_predictions)

# Add both the original 'label' and 'predicted_label' to the DataFrame
new_data_df['predicted_label'] = new_predictions

# Print the new data with both original and predicted labels
print("\nNew data with original and predicted labels:")
print(new_data_df)
