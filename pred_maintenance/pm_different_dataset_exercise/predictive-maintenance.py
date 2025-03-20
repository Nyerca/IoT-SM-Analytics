"""
1. UID: unique identifier ranging from 1 to 10000
2. product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
3. type: just the product type L, M or H from column 2
4. air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
5. process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
6. rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise
7. torque [Nm]: torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values.
8. tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
9. a 'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.

The machine failure consists of five independent failure modes

1. tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 - 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
2. heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tools rotational speed is below 1380 rpm. This is the case for 115 data points.
3. power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
4. overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
5. random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.

If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail.
This dataset is part of the following publication, please cite when using this dataset:
S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.
"""
# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, SVMSMOTE,RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

def show_distribution(df):

    # Identify categorical and numeric columns
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    numeric_cols = list(df.select_dtypes(include=['number']).columns)

    # Convert numeric columns with ≤3 unique values into categorical
    for col in numeric_cols[:]:  # Copy of list to avoid modifying while iterating
        if df[col].nunique() <= 3:
            categorical_cols.append(col)
            numeric_cols.remove(col)

    # Set up subplots dynamically
    num_cols = len(numeric_cols) + len(categorical_cols)
    rows = (num_cols // 3) + 1  # 3 plots per row

    fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axes = axes.flatten()  # Flatten axes for easier iteration

    # Plot numeric columns as histograms
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='blue', bins=30)
        axes[i].set_title(f"Distribution of {col}")

    # Plot categorical columns (including numeric ones with ≤3 unique values)
    for i, col in enumerate(categorical_cols, start=len(numeric_cols)):
        sns.countplot(x=df[col], ax=axes[i], palette='viridis', order=df[col].value_counts().index)
        axes[i].set_title(f"Class Distribution of {col}")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)


    for j in range(i + 1, len(axes)): # Hide empty subplots in the unused slots
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def show_outliers(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=['number']))
    plt.xticks(rotation=90)
    plt.title("Boxplot of Numerical Features (Outlier Detection)")
    plt.show()

def backward_regression(X, y, initial_list=[], threshold_out=0.5, verbose=True):
    """To select feature with Backward Stepwise Regression

    Args:
        X -- features values
        y -- target variable
        initial_list -- features header
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")

def analyze_model(y_test, y_predictions, model, model_name):
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions, average='weighted')
    precision = precision_score(y_test, y_predictions, average='weighted')
    f1s = f1_score(y_test, y_predictions, average='weighted')
    MCC = matthews_corrcoef(y_test, y_predictions)
    ROC_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='weighted')

    print(f"******************* {model_name} *******************")
    # Print Results
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1-Score: {f1s:.2%}")
    print(f"MCC: {MCC:.2%}")
    print(f"ROC AUC score: {ROC_AUC:.2%}")
    print(f"Time to train: {end_train - start:.2f} s")
    print(f"Time to predict: {end_predict - end_train:.2f} s")
    print(f"Total time: {end_predict - start:.2f} s")

    return accuracy, recall, precision, f1s,MCC, ROC_AUC

def plot_confusion_matrix(y_test, y_predictions, model_name, ax, show : False):


    # Assuming y_test and y_predictions are defined
    cm = confusion_matrix(y_test, y_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)  # Use existing subplot axis

    ax.set_title(f"{model_name}")  # Set title for that subplot

    if show:
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()



def plot_confusion_matrix_old(y_test, y_predictions):
    cm = confusion_matrix(y_test, y_predictions)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(5, 5))  # Create figure & axis once
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)  # Use existing axis

    plt.title("Confusion Matrix")
    plt.show()
# EDA


pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)

"""
It seems that there are two indices: the index and ProductID. 
We can drop those. 
There is a Type which is categorical and the remainder are numeric.
 The last five feastures are all failure modes, so they will not be evaluated in this notebook. 
 Drop the indices as these have no predictive power
"""

print("*********** DF ***********")
df = pd.read_csv('sensor_data.csv')
print(df)
df = df.drop(columns=['UDI'])
df = df.drop(columns=['Product ID'])

"""
Drop the failure modes, as we're only interested whether something is a failure. 
I guess that you'll build a model for each failure mode if it comes down to that.
"""

df.drop(['TWF','HDF','PWF','OSF','RNF'],axis=1,inplace=True)

print("*********** DF info ***********")
print(df.info())

print("*********** DF describe ***********")
print(df.describe(include='all').T)

"""
making sure that there are no missing values hidden as a question mark
"""


print("*********** DF replaced ***********")
df.replace("?",np.nan,inplace=True)
print(df)

"""
turn all columns into float to make processing later easier
"""

for column in df.columns:
    try:
        df[column]=df[column].astype(float)
    except:
        pass


"""
just check the descriptions for the numeric features. None missing and no apparent outliers
"""



print("*********** DF numeric ***********")
df_numeric = df.select_dtypes(include=[np.number])
print(df_numeric.describe(include='all').T)

"""
Another verification whether there are any missing features. I see none. 
"""

plt.figure(figsize=(15,15))
plot_kws={"s": 1}
sns.heatmap(df.isna().transpose(),
            cmap='cividis',
            linewidths=0.0,
            ).set_facecolor('white')

plt.figure(figsize=(10,10))
threshold = 0.80
sns.set_style("whitegrid", {"axes.facecolor": ".0"})



# CORRELATION

df_cluster2 = df_numeric.corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster2,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2,
            linecolor='lightgrey').set_facecolor('white')
plt.show()



"""
The profiling report follows to look for outliers, missing values, and distributions. We can see that the data is imbalanced. 
"""



show_distribution(df)


# DETECT OUTLIERS


show_outliers(df)


print(f"*********** Count of missing values per column *********** {df.isnull().sum()}")

"""
Drop the type, as this dominates too strongly on type = L.
"""

df.drop(['Type'],axis=1,inplace=True)

print("*********** list(df) ***********")
print(list(df))


print("*********** Turn categorical to numeric ***********")

df = pd.get_dummies(df,drop_first=True)
features = list(df.columns)
print(df)





"""
Just another confirmation of how badly imbalanced the data is. We'll need to oversample in this case to get a better prediction.
"""

print("*********** df_group machine failure ***********")
df_group = df.groupby(['Machine failure'])
print(df_group.count())


print("*********** Machine failure na ***********")
print(df[df['Machine failure'].isna()])

"""
Perform a statistical univariate test to determine the best features. Product type L dominates this strongly.
"""


# Feature Selection
best_features = SelectKBest(score_func=chi2, k='all')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
fit = best_features.fit(X, y)

# Creating DataFrame for Scores
df_scores = pd.DataFrame(fit.scores_, columns=["score"])
df_col = pd.DataFrame(X.columns, columns=["feature"])
feature_score = pd.concat([df_col, df_scores], axis=1)

# Sorting by Score
feature_score = feature_score.sort_values(by="score", ascending=True)

# Plot using Matplotlib
plt.figure(figsize=(10, 12))
plt.barh(feature_score["feature"].iloc[-20:], feature_score["score"].iloc[-20:], color="skyblue")
plt.xlabel("Score")
plt.ylabel("Feature")
plt.title("Top 20 Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)

plt.show()





Selected_Features = []


# Application of the backward regression function on our training data
backward_regression(X, y)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

feature_names = list(X.columns)


print(f"*********** shape of X *********** {np.shape(X)}")
print(f"*********** len(feature_names) *********** {len(feature_names)}")





########################################################## Modelling and Evaluation
"""
The data is too imbalanced, therefore we usually oversample.
 The randomoversampler performs better than SMOTE, but not oversampling performs the best. 
 This is curious, 
 but there are studies (check my discussion here: https://www.kaggle.com/competitions/autismdiagnosis/discussion/322588) 
 that reckon that it is better not to oversample.
"""



oversamp = RandomOverSampler(random_state=0)
# oversamp = SMOTE(n_jobs=-1)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify=y)
# X_train,y_train = oversamp.fit_resample(X_train, y_train)

"""
There are no distinct outliers, therefore a simple minmax scaler suffices. 
"""


sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','MCC score','time to train','time to predict','total time'])


sns.set_style("white")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Create a 2x3 grid

########################################################## Logistical Classification
start = time.time()
model = LogisticRegression().fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

model_name = "Logistical Classification"
accuracy, recall, precision, f1s,MCC,_ = analyze_model(y_test, y_predictions, model, model_name)
model_performance.loc[model_name] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]
plot_confusion_matrix(y_test, y_predictions, model_name, axes[0, 0],False)

########################################################## Decision Tree
from sklearn.tree import DecisionTreeClassifier
start = time.time()
model = DecisionTreeClassifier().fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_name = "Decision Tree"
accuracy, recall, precision, f1s,MCC,_ = analyze_model(y_test, y_predictions, model, model_name)
model_performance.loc[model_name] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]
plot_confusion_matrix(y_test, y_predictions, model_name,axes[0, 1],False)


print("*********** Feature importance:")
#plt.rcParams['figure.figsize']=10,10
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
print(feat_importances)
#feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
#sns.despine()
#plt.show()

########################################################## Random Forest
from sklearn.ensemble import RandomForestClassifier
start = time.time()
model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0,bootstrap=True,).fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_name = "Random Forest"
accuracy, recall, precision, f1s,MCC,_ = analyze_model(y_test, y_predictions, model, model_name)
model_performance.loc[model_name] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]
plot_confusion_matrix(y_test, y_predictions, model_name,axes[0, 2],False)

########################################################## Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
start = time.time()
model = GradientBoostingClassifier().fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_name = "Gradient Boosting Classifier"
accuracy, recall, precision, f1s,MCC,_ = analyze_model(y_test, y_predictions, model, model_name)
model_performance.loc[model_name] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]
plot_confusion_matrix(y_test, y_predictions, model_name,axes[1, 0],False)


print("*********** Feature importance:")
#plt.rcParams['figure.figsize']=10,10
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
print(feat_importances)
#feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
#sns.despine()
#plt.show()

########################################################## Neural Network MLP
from sklearn.neural_network import MLPClassifier

start = time.time()
model = MLPClassifier(hidden_layer_sizes = (100,100,),
                      activation='relu',
                      solver='adam',
                      batch_size=2000,
                      verbose=0).fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_name = "Neural Network MLP"
accuracy, recall, precision, f1s,MCC,_ = analyze_model(y_test, y_predictions, model, model_name)
model_performance.loc[model_name] = [accuracy, recall, precision, f1s,MCC,end_train-start,end_predict-end_train,end_predict-start]
plot_confusion_matrix(y_test, y_predictions, model_name,axes[1, 1],True)

########################################################## Conclusion
model_performance.fillna(.90,inplace=True)
print(model_performance)



"""
The bulk of the prediction lays with the type Product L that strongly suggests a failure. 
This however only gave a 90% accuracy. 
I dropped the type and got much more accurate results as a result. 
Random forest seems to be the go-to model for this case, giving even a decent MCC score. 
"""

