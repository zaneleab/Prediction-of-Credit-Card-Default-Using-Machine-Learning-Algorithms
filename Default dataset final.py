#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')


# In[3]:


df #displays the dataframe


# In[4]:


print(df.columns)


# In[5]:


print('Number of rows having all values as null:')
print(df.isnull().all(axis=1).sum())
# Finding number of rows through sum function which have missing values


# In[6]:


# Finding the number of columns through sum function which have missing values

print('Number of columns having all values as null:')
print(df.isnull().all(axis=0).sum())


# In[7]:


df.info() #shows me any null values, in this case, i do not have any null values, the dataset does not have any missing or incomplete values.


# In[8]:


df.describe() #summary statistics for numeric columns, helps me to identify any missing data


# In[9]:


# Check data types of all columns after conversion
for column in df.columns:
    data_type = df[column].dtype
    print(f"Column: {column}, Data Type: {data_type}")


# In[10]:


for column in df.columns:
    unique_values = df[column].unique()
    print(f"Column: {column}, Unique Values: {unique_values}")


# In[11]:


sns.heatmap(df.isnull(), cbar=False, cmap='viridis') #creates a heatmap using seaborn. The missing_data DataFrame is used as input, and the 'viridis' color map is specified
plt.title('Missing Data Heatmap')
plt.xlabel('rows')
plt.ylabel('columns')
plt.show() # heatmap displaying missing values as yellow, making it easy to identify where data is missing.My heatmap shows that there is no missing data in my dataset.


# In[12]:


#Handling outliers


# In[13]:


# Set the Z-Score threshold for outliers
z_score_threshold = 3

# Create an empty list to store outlier points
outlier_points = []

# Iterate through all columns
for column in df.columns:
    # Check for outliers using Z-Score
    z_scores = stats.zscore(df[column])
    outliers = (z_scores > z_score_threshold) | (z_scores < -z_score_threshold)
    
    # Get the indices of outlier rows
    outlier_indices = np.where(outliers)[0]
    
    # Add the (column, outlier_indices) pair to the outlier_points list
    outlier_points.append((column, outlier_indices))
    
    # Check if there are outliers for this column
    if len(outlier_indices) > 0:
        print(f"Outliers detected in column: {column}, Count: {len(outlier_indices)}")

# Create a scatterplot for each pair of (column, outlier_indices)
for column, outlier_indices in outlier_points:
    # Extract the data for the scatterplot
    x = df[column].iloc[outlier_indices]
    y = [column] * len(outlier_indices)
    
    # Create the scatterplot
    plt.scatter(x, y, label=column)

# Set plot labels and title
plt.xlabel("Outlier Values")
plt.ylabel("Columns")
plt.title("Scatterplot of Outliers Detected by Z-Score")

# Show the legend
plt.legend()

# Show the scatterplot
plt.show()


# In[14]:


default_df = df.copy()


# In[15]:


#Categorical and continuous columns
categorical_columns =[]
continuous_columns = []
for columns in default_df.columns:
    if default_df[columns].nunique() > 12:
        continuous_columns.append(columns)
    else:
        categorical_columns.append(columns)
continuous_columns


# In[16]:


continuous_columns


# In[17]:


categorical_columns


# In[18]:


fig, axes = plt.subplots(3, 3, figsize=(20, 15))

for i, continuous_col in enumerate(continuous_columns[:min(9, len(continuous_columns))]):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    sns.boxenplot(data=default_df, y=continuous_col, x='Default', hue='Default', ax=ax)
    ax.set_title(f'{continuous_col} vs. Default')

# Hide any unused subplots
for i in range(len(continuous_columns), len(axes.flat)):
    axes.flatten()[i].axis('off')

plt.show()


# In[19]:


# Define a dictionary to store the outlier information
outliers_info = {
    'Loan': 1,
   
}

# Create a dictionary to store quantile information for each column before and after cleaning
quantiles_info_before = {}
quantiles_info_after = {}

# Iterate through columns and process outliers
for column, count in outliers_info.items():
    # Calculate quantiles before cleaning
    quantiles_before = df[column].quantile([0.5, 0.75, 0.90, 0.95, 0.99])
    quantiles_info_before[column] = quantiles_before
    
    if count > 0:
        # Remove outliers by keeping only values within the specified quantile range
        lower_quantile = df[column].quantile(0.01)
        upper_quantile = df[column].quantile(0.99)
        df = df[(df[column] >= lower_quantile) & (df[column] <= upper_quantile)]
    
    # Calculate quantiles after cleaning
    quantiles_after = df[column].quantile([0.5, 0.75, 0.90, 0.95, 0.99])
    quantiles_info_after[column] = quantiles_after

# Print the quantile information before and after for each column with labels
for column in outliers_info.keys():
    print(f"Quantile Information for Column: {column}")
    
    # Before cleaning
    print("Before Cleaning:")
    quantiles_before = quantiles_info_before[column]
    print(f"50th Percentile: {quantiles_before[0.5]}")
    print(f"75th Percentile: {quantiles_before[0.75]}")
    print(f"90th Percentile: {quantiles_before[0.90]}")
    print(f"95th Percentile: {quantiles_before[0.95]}")
    print(f"99th Percentile: {quantiles_before[0.99]}")
    
    # After cleaning
    print("After Cleaning:")
    quantiles_after = quantiles_info_after[column]
    print(f"50th Percentile: {quantiles_after[0.5]}")
    print(f"75th Percentile: {quantiles_after[0.75]}")
    print(f"90th Percentile: {quantiles_after[0.90]}")
    print(f"95th Percentile: {quantiles_after[0.95]}")
    print(f"99th Percentile: {quantiles_after[0.99]}")
    
    print()


# In[20]:


sns.histplot(
    default_df['Age'] , kde=True, bins = 50
)


# In[21]:


#Handling data imbalance 


# In[22]:


sns.countplot(data = default_df, x='Default') #large imbalnce in the data. More than double of the people in our dataset will not default. data is not distributed equally.


# In[23]:


# Preserve original dataset 
df_smote = df.copy()


# In[24]:


df_smote


# In[25]:


sns.scatterplot(
    data = default_df, x='Age',y = 'Loan to Income', hue= 'Default',
) #plot shows data is massively skewed in favour of non-defaulters


# In[26]:


oversample = SMOTE()
X_input , y_output = df_smote.iloc[:,:-1],df_smote[['Default']]
X,y = oversample.fit_resample(X_input,y_output)
print('Shape of X {}'.format(X.shape))
print('Shape of y {}'.format(y.shape))
df_smote = pd.concat([X,y],axis=1)
print('Normal distributed dataset shape {}'.format(df_smote.shape))


# In[27]:


fig, ax = plt.subplots(1,2,figsize=(15,10))
sns.scatterplot(
    data = df, x='Age',y = 'Loan to Income', hue= 'Default',ax=ax[0],
)
ax[0].set_title('UNBALANCED DATASET')
sns.scatterplot(
    data = df_smote, x = 'Age',y = 'Loan to Income' , hue = 'Default',ax=ax[1]
)
ax[1].set_title('BALANCED DATASET')


# In[28]:


df_smote


# In[74]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Separate features (X) and target variable (y)
X = df_smote.drop('will_default', axis=1)
y = df_smote['will_default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection using SelectKBest and f_classif
k = 4  # Set the desired number of features
k_best = SelectKBest(score_func=f_classif, k=k)
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# Display selected features
selected_features = X.columns[k_best.get_support()]
print("Selected Features:", selected_features)

# Define the neural network model
def create_nn_model(optimizer='adam', activation='relu', neurons=64):
    nn_model = Sequential([
        Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation),
        Dense(32, activation=activation),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return nn_model

# Specify hyperparameter grid for the neural network
param_grid_nn = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'sigmoid'],
    'neurons': [32, 64, 128]
}

# Create a GridSearchCV object for the neural network
grid_search_nn = GridSearchCV(create_nn_model, param_grid_nn, cv=5, scoring='accuracy', refit=True)
grid_search_nn.fit(X_train_scaled, y_train)

# Get the best neural network model
nn_model_best = grid_search_nn.best_estimator_

# Train the best neural network model
nn_model_best.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate all models
confusion_matrices = {}  # Dictionary to store confusion matrices

for algorithm, model in classifiers.items():
    if algorithm == 'NeuralNetwork':
        y_pred = (nn_model_best.predict(X_test_scaled) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_selected)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[algorithm] = cm
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.show()

    # Other metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{algorithm} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print()


# In[83]:


# Lists to store metric values
algorithms = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Iterate through classifiers
for algorithm, model in classifiers.items():
    if algorithm == 'NeuralNetwork':
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_selected)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append values to lists
    algorithms.append(algorithm)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Bar chart for accuracy
plt.figure(figsize=(10, 5))
plt.bar(algorithms, accuracies, color='skyblue')
plt.title('Model Comparison - Accuracy')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()



# Bar chart for precision
plt.figure(figsize=(10, 5))
plt.bar(algorithms, precisions, color='lightcoral')
plt.title('Model Comparison - Precision')
plt.xlabel('Algorithms')
plt.ylabel('Precision')
plt.ylim(0, 1)
plt.show()

# Bar chart for recall
plt.figure(figsize=(10, 5))
plt.bar(algorithms, recalls, color='lightgreen')
plt.title('Model Comparison - Recall')
plt.xlabel('Algorithms')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.show()

# Bar chart for F1-Score
plt.figure(figsize=(10, 5))
plt.bar(algorithms, f1_scores, color='gold')
plt.title('Model Comparison - F1-Score')
plt.xlabel('Algorithms')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.show()


# In[ ]:




