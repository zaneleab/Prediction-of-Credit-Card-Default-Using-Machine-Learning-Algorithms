#!/usr/bin/env python
# coding: utf-8

# In[92]:


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


# In[93]:


excel_file = 'default of credit card clients.xls' #path to my Excel file


# In[94]:


df = pd.read_excel(excel_file) #reads excel file as dataframe


# In[4]:


df #displays the dataframe


# In[5]:


print(df.columns)


# In[6]:


df = df.rename(columns={
    'Unnamed: 0': 'ID',
    'X1': 'Balance_Limit',
    'X2': 'Sex',
    'X3': 'Education',
    'X4': 'Marriage',
    'X5': 'Age',
    'X6': 'repayment_sept',
    'X7': 'repayment_aug',
    'X8': 'repayment_july',
    'X9': 'repayment_june',
    'X10': 'repayment_may',
    'X11': 'repayment_april',
    'X12': 'bill_sept',
    'X13': 'bill_aug',
    'X14': 'bill_july',
    'X15': 'bill_june',
    'X16': 'bill_may',
    'X17': 'bill_april',
    'X18': 'previous_payment_sept',
    'X19': 'previous_payment_aug',
    'X20': 'previous_payment_july',
    'X21': 'previous_payment_june',
    'X22': 'previous_payment_may',
    'X23': 'previous_payment_april',
    'Y': 'will_default'
})


# In[7]:



# Drop row with index label 0
df.drop(0, axis=0, inplace=True)


# In[8]:


df


# In[9]:


taiwan_df = df.copy()


# In[10]:


df


# In[11]:


print('Number of rows having all values as null:')
print(df.isnull().all(axis=1).sum())
# Finding number of rows through sum function which have missing values


# In[12]:


df.info() #shows me any null values, in this case, i do not have any null values, the dataset does not have any missing or incomplete values.


# In[13]:


df.describe() #summary statistics for numeric columns, helps me to identify any missing data


# In[14]:


print(df.duplicated(subset=None,keep='first').count())

# The column used for checking is id column
duplicate_rows=taiwan_df[taiwan_df.duplicated(['ID'])]
print("duplicate_rows :", duplicate_rows) #there are no duplicate rows in my dataset


# In[15]:


df


# In[16]:



# Check data types of all columns after conversion
for column in df.columns:
    data_type = df[column].dtype
    print(f"Column: {column}, Data Type: {data_type}")


# In[17]:


for column in df.columns:
    unique_values = df[column].unique()
    print(f"Column: {column}, Unique Values: {unique_values}") #check there are no duplicated columns by checking for the number of unique columns


# In[18]:



df['ID'] = df['ID'].astype(int)
df['Balance_Limit'] = df['Balance_Limit'].astype(int)
df['Sex'] = df['Sex'].astype(int)
df['Education'] = df['Education'].astype(int)
df['Marriage'] = df['Marriage'].astype(int)
df['Age'] = df['Age'].astype(int)
df['repayment_sept'] = df['repayment_sept'].astype(int)
df['repayment_aug'] = df['repayment_aug'].astype(int)
df['repayment_july'] = df['repayment_july'].astype(int)
df['repayment_june'] = df['repayment_june'].astype(int)
df['repayment_may'] = df['repayment_may'].astype(int)
df['repayment_april'] = df['repayment_april'].astype(int)
df['bill_sept'] = df['bill_sept'].astype(int)
df['bill_aug'] = df['bill_aug'].astype(int)
df['bill_july'] = df['bill_july'].astype(int)
df['bill_june'] = df['bill_june'].astype(int)
df['bill_may'] = df['bill_may'].astype(int)
df['bill_april'] = df['bill_april'].astype(int)
df['previous_payment_sept'] = df['previous_payment_sept'].astype(int)
df['previous_payment_aug'] = df['previous_payment_aug'].astype(int)
df['previous_payment_july'] = df['previous_payment_july'].astype(int)
df['previous_payment_june'] = df['previous_payment_june'].astype(int)
df['previous_payment_may'] = df['previous_payment_may'].astype(int)
df['previous_payment_april'] = df['previous_payment_april'].astype(int)
df['will_default'] = df['will_default'].astype(int)


# In[19]:



sns.heatmap(df.isnull(), cbar=False, cmap='viridis') #creates a heatmap using seaborn. The missing_data DataFrame is used as input, and the 'viridis' color map is specified
plt.title('Missing Data Heatmap')
plt.xlabel('rows')
plt.ylabel('columns')
plt.show() # heatmap displaying missing values as yellow, making it easy to identify where data is missing.My heatmap shows that there is no missing data in my dataset.


# In[20]:


#Handling outliers


# In[21]:


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


# In[22]:


#when removing outliers, after analysing the data, i have made the decision not to remove outliers for the variables balance limit, Education and Age as these provide valuableinformation for my research.


# In[23]:


taiwan_df = df.copy()


# In[24]:


#Categorical and continuous columns
categorical_columns =[]
continuous_columns = []
for columns in taiwan_df.columns:
    if taiwan_df[columns].nunique() > 12:
        continuous_columns.append(columns)
    else:
        categorical_columns.append(columns)
continuous_columns


# In[25]:


continuous_columns


# In[26]:


categorical_columns


# In[27]:



fig, axes = plt.subplots(3, 3, figsize=(20, 15))

for i, categorical_col in enumerate(categorical_columns[:min(9, len(categorical_columns))]):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    sns.boxenplot(data=taiwan_df, y='Balance_Limit', x=categorical_col, hue='will_default', ax=ax)
    ax.set_title(f'{categorical_col} vs. Balance_Limit')

plt.show()


# In[28]:


fig, axes = plt.subplots(3, 3, figsize=(20, 15))

for i, categorical_col in enumerate(categorical_columns[:min(9, len(categorical_columns))]):
    row, col = i // 3, i % 3
    ax = axes[row, col]

    sns.boxenplot(data=taiwan_df, y='Age', x=categorical_col, hue='will_default', ax=ax)
    ax.set_title(f'{categorical_col} vs. Age')

plt.show()


# In[29]:


# Create a dictionary to map values in the 'Marriage' column to labels
marriage_labels = {1: 'Married', 2: 'Single', 3: 'Other'}

# Replace the numeric values in the 'Marriage' column with labels
taiwan_df['Marriage'] = taiwan_df['Marriage'].map(marriage_labels)

# Create a FacetGrid with 'will_default' as rows and 'Marriage' as columns
g = sns.FacetGrid(taiwan_df, row='will_default', col='Marriage')

# Plot histograms of 'Age' in each facet
g = g.map(plt.hist, 'Age')

# Show the plot
plt.show()


# In[30]:


unique_marriage_values = taiwan_df['Marriage'].unique()
print(unique_marriage_values)


# In[31]:


"""From above plot we can infer that married people between age bracket of 30 and 50 and unmarried clients of age 20-30 tend to default payment with unmarried clients higher probability to default payment. Hence we can include MARRIAGE feature of clients to find probability of defaulting the payment next month'"""


# In[32]:


sex_labels = {1: 'Male', 2: 'Female'}

# Replace the numeric values in the 'Sex' column with labels
taiwan_df['Sex'] = taiwan_df['Sex'].map(sex_labels)

# Create a FacetGrid with 'will_default' as rows and 'Sex' as columns
g = sns.FacetGrid(taiwan_df, row='will_default', col='Sex')

# Create a bar plot to visualize the distribution of 'Sex'
g = g.map(sns.countplot, 'Sex')

# Show the plot
plt.show()


# In[33]:


"""It can be seen that females of age group 20-30 have very high tendency to default payment compared to males in all age brackets. Hence we can keep the SEX column of clients to predict probability of defaulting payment."""


# In[34]:


# Define a dictionary to store the outlier information
outliers_info = {
    'repayment_sept': 141,
    'repayment_aug': 157,
    'repayment_july': 150,
    'repayment_june': 169,
    'repayment_may': 164,
    'repayment_april': 129,
    'bill_sept': 686,
    'bill_aug': 670,
    'bill_july': 661,
    'bill_june': 680,
    'bill_may': 651,
    'bill_april': 651,
    'previous_payment_sept': 390,
    'previous_payment_aug': 307,
    'previous_payment_july': 362,
    'previous_payment_june': 396,
    'previous_payment_may': 414,
    'previous_payment_april': 439
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


# In[35]:


df


# In[36]:



"""
In this column, there are values outside of the define keys 1 - Graduate school
2 - University
3 - High school
4 - others
5 - Unknown
To resolve this problem, i have changed any answers out of the defined key to unknown """

# Define a custom function to replace values outside the range 1-5 with 5
def replace_values(value):
    if 1 <= value <= 5:
        return value
    else:
        return 5

# Apply the custom function to the 'Education' column
taiwan_df['Education'] = taiwan_df['Education'].apply(replace_values).astype(int)


# In[37]:



# First, i create a count of values in the 'Marriage' column.
education_counts = taiwan_df['Education'].value_counts()

# Create labels and values for the pie chart
labels = education_counts.index.tolist()  # Get the unique categories as labels
values = education_counts.values

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot a pie chart on the first subplot (ax[0])
ax[0].pie(values, labels=labels, autopct='%1.1f%%', shadow=True)

# Plot a countplot on the second subplot (ax[1])
sns.countplot(data=taiwan_df, x='Education', ax=ax[1])

# Set labels for the x-axis ticks in the countplot)
ax[1].set_xticklabels(['Graduate school','University', 'High school', 'Others', 'Unknown'])

# Add a title for the countplot
ax[1].set_title('Education Distribution')

# Adjust the layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()


# In[38]:


taiwan_df['Sex'].value_counts()
labels = ['female','male']
values = taiwan_df['Sex'].value_counts().values

fig, ax = plt.subplots(1,2)
ax[0].pie(values, labels = labels, autopct='%1.1f%%', shadow = True)
sns.countplot(
    data = taiwan_df, x = 'Sex', ax=ax[1]
)
plt.tight_layout()
plt.show() #a histogram to show the imbalance of data between sexes 


# In[39]:



# First, let's create a count of values in the 'Marriage' column.
marriage_counts = taiwan_df['Marriage'].value_counts()

# Create labels and values for the pie chart
labels = marriage_counts.index.tolist()  # Get the unique categories as labels
values = marriage_counts.values

# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot a pie chart on the first subplot (ax[0])
ax[0].pie(values, labels=labels, autopct='%1.1f%%', shadow=True)

# Plot a countplot on the second subplot (ax[1])
sns.countplot(data=taiwan_df, x='Marriage', ax=ax[1])

# Set labels for the x-axis ticks in the countplot (ensure all unique categories are included)
ax[1].set_xticklabels(['Married', 'Single', 'Others'])

# Add a title for the countplot
ax[1].set_title('Marriage Distribution')

# Adjust the layout for better visualization
plt.tight_layout()

# Show the plots
plt.show()


# In[40]:


sns.histplot(
    taiwan_df['Age'] , kde=True, bins = 50
)


# In[41]:


taiwan_df


# In[42]:


#Handling data imbalance 


# In[43]:


sns.countplot(data = taiwan_df, x='will_default') #large imbalnce in the data. More than double of the people in our dataset will not default. data is not distributed equally.


# In[44]:


# Preserve original dataset 
df_smote = df.copy()


# In[45]:


df_smote


# In[46]:


sns.scatterplot(
    data = taiwan_df, x='bill_sept',y = 'Balance_Limit', hue= 'will_default',
) #plot shows data is massively skewed in favour of non-defaulters


# In[47]:


# Convert 'will_default' to integer type
df_smote['will_default'] = df_smote['will_default'].astype(int)


# In[48]:


oversample = SMOTE()
X_input , y_output = df_smote.iloc[:,:-1],df_smote[['will_default']]
X,y = oversample.fit_resample(X_input,y_output)
print('Shape of X {}'.format(X.shape))
print('Shape of y {}'.format(y.shape))
df_smote = pd.concat([X,y],axis=1)
print('Normal distributed dataset shape {}'.format(df_smote.shape))


# In[49]:


fig, ax = plt.subplots(1,2,figsize=(15,10))
sns.scatterplot(
    data = df, x='bill_sept',y = 'Balance_Limit', hue= 'will_default',ax=ax[0],
)
ax[0].set_title('UNBALANCED DATASET')
sns.scatterplot(
    data = df_smote, x = 'bill_sept',y = 'Balance_Limit' , hue = 'will_default',ax=ax[1]
)
ax[1].set_title('BALANCED DATASET')


# In[50]:


#the aove plot shows the highest correlation between default and repayments in all months


# In[51]:


# Separate features (X) and target variable (y)
X = df_smote.drop('will_default', axis=1)
y = df_smote['will_default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[52]:


from sklearn.feature_selection import SelectKBest, f_classif

# Perform feature selection using SelectKBest and f_classif
k = 10  # Set the desired number of features
k_best = SelectKBest(score_func=f_classif, k=k)
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# Display selected features
selected_features = X.columns[k_best.get_support()]
print("Selected Features:", selected_features)


# In[53]:


df_smote['Dues']= df_smote['bill_april']+ df_smote['bill_may'] + df_smote['bill_june']+ df_smote['bill_july']+ df_smote['bill_aug'] + df_smote['bill_sept']
df_smote['Previous_payments'] = df_smote['previous_payment_april'] + df_smote['previous_payment_may'] + df_smote['previous_payment_june'] + df_smote['previous_payment_july'] + df_smote['previous_payment_aug'] + df_smote['previous_payment_sept']


# In[54]:


df_smote_grouped = df_smote.groupby('will_default')['Dues'].mean()
label = ['Non Defaulter', 'Defaulter']
values = df_smote_grouped.values

fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Create a 1x2 grid of subplots

# Subplot 1: Bar plot
ax[0].bar(label, values)
ax[0].set_title('Bill Dues of Non-Defaulter Vs Defaulter')
ax[0].set_ylabel('Total Bills (from April to September)')

# Subplot 2: Histogram
sns.histplot(data=df_smote, x='Dues', hue='will_default', kde=True, ax=ax[1])
ax[1].set_title('Distribution of Bill Dues')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[55]:


df_smote.groupby('will_default')['Previous_payments'].mean()
label = ['Non Defaulter','Defaulter']
values = df_smote.groupby('will_default')['Previous_payments'].mean().values
plt.bar(label, values)
plt.title('Previous Payments\n[Non-Defaulter Vs Defaulter]')
plt.ylabel('Previous Payments(from april to september)')
plt.tight_layout


# In[56]:


df_smote


# In[57]:


df_smote.drop(['ID','bill_sept','bill_aug','bill_july','bill_june','bill_may','bill_april','previous_payment_sept','previous_payment_aug','previous_payment_july','previous_payment_june','previous_payment_may','previous_payment_may','previous_payment_april'],axis=1, inplace=True)


# In[58]:


df_smote


# In[91]:


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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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

# Function to create a Keras model for GridSearchCV
def create_nn_model():
    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a dictionary of classifiers
classifiers = {
    'SupportVectorMachine': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'NeuralNetwork': Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

# Train the models
confusion_matrices = {}  # Dictionary to store confusion matrices


# Define hyperparameter grids for each classifier
param_grids = {
    'SupportVectorMachine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'base_estimator__max_depth': [1, 2, 4]
    },
    'NeuralNetwork': {'epochs': [30, 50, 100], 'batch_size': [16, 32, 64]}
}

# Train the models with hyperparameter tuning
nn_model_best = None  # Initialize nn_model_best outside the try block

for algorithm, model in classifiers.items():
    try:
        if algorithm != 'NeuralNetwork':
            param_grid = param_grids[algorithm]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_selected, y_train)
            classifiers[algorithm] = grid_search.best_estimator_
        else:
            param_grid_nn = param_grids[algorithm]
            nn_model = KerasClassifier(build_fn=create_nn_model, verbose=0)
            grid_search_nn = GridSearchCV(nn_model, param_grid_nn, cv=5, scoring='accuracy', refit=True)
            grid_search_nn.fit(X_train_scaled, y_train)
            nn_model_best = grid_search_nn.best_estimator_.model  # Corrected assignment
            nn_model_best.fit(X_train_scaled, y_train)  # Fit the NeuralNetwork here

    except Exception as e:
        print(f"Error in {algorithm}: {e}")
        print("Neural Network model could not be trained.")


# Ensure that Neural Network is successfully trained before making predictions
if nn_model_best is not None:
    y_pred = (nn_model_best.predict(X_test_scaled) > 0.5).astype(int)
    
# Ensure that AdaBoostClassifier is fitted before making predictions
classifiers['AdaBoostClassifier'].fit(X_train_selected, y_train)

for algorithm, model in classifiers.items():
    if algorithm == 'NeuralNetwork':
        try:
            y_pred = (nn_model_best.predict(X_test_scaled) > 0.5).astype(int)
        except Exception as e:
            print(f"Error in NeuralNetwork prediction: {e}")
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


# #Optimisation of mode

# In[ ]:





# 
