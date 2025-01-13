# Diabetes Prediction using Random Forest

1.2 Code Implementation

1.2.1 Import libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
```
pandas: is used for data manipulation and analysis.
numpy: is a library for numerical computations in Python.
matplotlib.pyplot: is a plotting library for creating static, animated, and interactive visualizations.
seaborn: is built on top of Matplotlib and is used for making statistical graphics in Python.
lightgbm.LGBMRegressor: LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework designed for speed and efficiency. for this study it was used for predicting missing values.
sklearn.model_selection.train_test_split: used to splits the dataset into training and testing subsets for model evaluation.
sklearn.model_selection.StratifiedKFold: This is a cross-validation technique from scikit-learn that splits the dataset into k subsets while ensuring each fold has the same proportion of each target class (useful for imbalanced datasets).
sklearn.metrics.accuracy_score: This function computes the accuracy of a classification model by comparing predicted and true labels.
sklearn.metrics.confusion_matrix: This function generates a confusion matrix, which shows the number of correct and incorrect predictions made by the model for each class
sklearn.metrics.classification_report: This function generates a report showing various classification metrics, such as precision, recall, F1-score, and support for each class.
sklearn.preprocessing.RobustScaler: This scaler is used to scale features using statistics that are robust to outliers, such as interquartile range.
sklearn.ensemble.RandomForestClassifier: This is an ensemble learning method that constructs multiple decision trees and combines their results to improve classification accuracy and reduce overfitting.
imblearn.over_sampling.SMOTE: SMOTE (Synthetic Minority Over-sampling Technique) is a method to address class imbalance by creating synthetic examples in the minority class.
warnings.filterwarnings('ignore'): This function suppresses warnings that might be displayed during the execution of the code.

1.2.2 Read and inspect data frame
```python
df = pd.read_csv('sample_data/diabetes_prediction_dataset.csv')
```

```python
#Displays the first 5 rows of the data frame
df.head()
```
```python
#dimensions of the data frame(100,000 rows, 9 columns)
df.shape 
```
```python
#generates a summary of statistics for the numerical columns in the data frame
df.describe()
```
```python
#check for missing (NaN) values in the data frame 
df.isna().sum()
```
```python
#identify and count the number of duplicate rows in the data frame
df.duplicated( keep=False ).sum()
```
```python
#extracts the rows that are duplicates from the data frame df and then displays the first few rows of the duplicate entries
duplicate_rows = df[ df.duplicated( keep=False ) ]
duplicate_rows.head(30)
```

1.2.3 Select Data Frame columns based on their data type

```python
#list of column names from df that are numeric types, excluding the last column (Diabetes)
df_continuous_columns = df.select_dtypes( include = np.number ).columns[ : -1 ]
df_continuous_columns
```
```python
#list of column names from df that are categorical types, excluding the last column
df_categorical_columns = df.select_dtypes( exclude = np.number ).columns
df_categorical_columns
```
1.2.4 Exploratory data analysis: Visualize distributions of the Data Frame

1.2.4.1 Display a count plot for the diabetes column
```python
sns.countplot(x='diabetes', data=df)
plt.show()
```


1.2.4.2 Display a side-by-side count plot for each categorical column
```python
figure, axs = plt.subplots( nrows=1, ncols=2, figsize=(20,10))
for i, column in enumerate( df_categorical_columns ):

    sns.countplot( x= df[column],  ax=axs[ i ] )
    axs[ i ].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.4.3 Display histograms for continuous columns
```python
figure, axs = plt.subplots( nrows=3, ncols=2, figsize=(20,20))
for i, column in enumerate( df_continuous_columns ):
    row = i // 2
    col = i % 2
    sns.histplot( df[column],  ax=axs[row, col] )
    axs[row, col].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.4.4 Display box plots for continuous columns
```python
figure, axs = plt.subplots( nrows=3, ncols=2, figsize=(20,20))
for i, column in enumerate( df_continuous_columns ):
    row = i // 2
    col = i % 2
    sns.boxplot( df[column],  ax=axs[row, col] )
    axs[row, col].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.4.4 Display pair plots for continuous columns
```python
sns.pairplot( df )
plt.show()
```

1.2.4.5 Identify relationships between variables
```python
sns.heatmap( df[ df.select_dtypes( include = np.number ).columns ].corr(), annot=True, cmap='Blues' )
plt.show()
```

1.2.5 Handling Outliers
1.2.5.1 Create a list of columns for outlier handling
```python
column_to_clean = [ 'bmi', 'HbA1c_level', 'blood_glucose_level' ]
```
Since not all columns contain outliers, we will only handle the columns that need cleaning.
```python
#Obtain the Q1 for each column, which will be used to handle outliers.
Q1 = df[ column_to_clean ].quantile( 0.25 )
Q1
```
```python
#Obtain the Q3 for each column, which will be used to handle outliers.
Q3 = df[ column_to_clean ].quantile( 0.75 )
Q3
```
```python
#Obtain the IQR for each column, which will be used to calculate the lower and upper bound for each column.
IQR = Q3 - Q1
IQR
```
```python
#calculates the lower bound for detecting outliers using the Interquartile Range (IQR) method
lower_bound = Q1 - 1.5 * IQR
lower_bound
```
```python
#calculates the upper bound for detecting outliers using the Interquartile Range (IQR) method
upper_bound = Q3 + 1.5 * IQR
upper_bound
```

1.2.5.2 Count the number of outliers in each column
```python
outliers_per_col = df[ (df[ column_to_clean ] < lower_bound) | (df[ column_to_clean ] > upper_bound) ].count()
outliers_per_col.sort_values( )
```
This will give us the count of outliers in each column and sort them in ascending order.
```python
#total count of outliers across all columns
outliers_per_col.sum()
```

1.2.5.3 Create a copy of the Data Frame
```python
new_df = df.copy()
```

1.2.5.4 Standardize the values in the smoking history column
```python
new_df.loc[ ( (new_df[ 'smoking_history' ] == 'not current' ) | ( new_df[ 'smoking_history' ] == 'ever' ) ), 'smoking_history' ] = 'former'
```
This operation allows you to standardize the values in the smoking history column by merging similar categories.

1.2.5.5 Replace outlier values with NaN
```python
for col in column_to_clean:
    new_df.loc[ ( (new_df[ col ] < lower_bound[ col ] ) | ( new_df[ col ] > upper_bound[ col ] ) ), col ] = np.nan

```
Replace outlier values in the specified columns with NaN, which can then be handled later (e.g., through imputation ).

1.2.6 Inspect the data frame
```python
#Displays the first 5 rows of the data frame
new_df.head()
```
```python
#generates a summary of statistics for the numerical columns in the DataFrame
new_df.describe()
```
```python
#check for missing (NaN) values in the data frame
new_df.isna().sum().sort_values()
```

1.2.7 Impute missing values using LightGBM to predict them
```python
def convert_to_numeric( continuous_df, categorical_columns, categorical, pred_col ):
    for column in categorical_columns:
      #we won't be converting the column we want to predict its NaN values for categorical predictions
      if column == pred_col and categorical == True:
        continue
      else:
        continuous_df[ column ] = pd.factorize( continuous_df[ column ] )[ 0 ]

    return continuous_df
```

The function, convert_to_numeric(), is designed to convert categorical columns in a data frame to numeric values using pd.factorize, with the exception of the target column you're predicting, which contains missing values.
Parameters:
continuous_df: The DataFrame containing the columns to be converted.
categorical_columns: A list of columns in continuous_df that are categorical.
categorical: A boolean indicating if the target column is categorical (used for condition).
pred_col: The name of the column you're trying to predict, which should not be converted if it's categorical.

Logic:
The loop iterates over all categorical_columns.
If the column is the target (pred_col) and is categorical, it skips the conversion for that column.
Otherwise, it uses pd.factorize() to convert the categorical values in the column to numeric labels. pd.factorize() assigns an integer value to each unique category.
Return The modified data frame(continuous_df) with converted columns is returned.

```python
def predict_missing_values( new_df, columns, pred_df, categorical_data, categorical_columns):
    dataframe = new_df.copy()
    for col in columns:
        # if more than 50% of the column data is missing, drop column
        if dataframe[col].isna().sum() > dataframe.shape[0] / 2:
            dataframe.drop(col, axis=1, inplace=True)
            pred_df.drop(col, axis=1, inplace=True) # Drop from pred_df as well
        else:
            # Get a boolean mask of missing values
            col_missing_mask = dataframe[col].isna()

            # No missing index
            if not col_missing_mask.any():  # Check if any values are missing
                continue
            else:
                # Create 'is_nan' column and set values based on the mask
                dataframe['is_nan'] = 0
                dataframe.loc[col_missing_mask, 'is_nan'] = 1

                continuous_df = dataframe.copy()

                continuous_df = convert_to_numeric(continuous_df, categorical_columns, categorical_data, col)

                train = continuous_df[continuous_df['is_nan'] == 0]
                test = continuous_df[continuous_df['is_nan'] == 1]

                X_train = train.drop([col, 'is_nan'], axis=1)
                y_train = train[col]

                X_test = test.drop([col, 'is_nan'], axis=1)
                y_test = test[col]

                lgbm = LGBMRegressor()
                lgbm.fit(X_train, y_train)

                y_pred = lgbm.predict(X_test)

                # Use the mask to assign values to pred_df
                pred_df.loc[col_missing_mask, col] = y_pred
```
The predict_missing_values() function is designed to predict missing values in the specified columns of a data frame using LightGBM (LGBMRegressor). If a column has more than 50% missing values, it is dropped. Otherwise, the function trains a LightGBM model to predict the missing values based on other features and fills in the missing entries in a new data frame.
Parameters:
new_df: The data frame with missing values to be imputed.
columns: A list of column names in new_df that may have missing values.
pred_df: The data framewhere the predicted values will be stored.
categorical_data: Boolean value indicating if the target column is categorical.
categorical_columns: A list of categorical columns to be considered for factorization.

Logic:
Check Missing Data: The function first checks if more than 50% of a column is missing. If so, that column is dropped from both data frame (a copy of new_df) and pred_df.
Handle Missing Values:

For columns with missing values, a boolean mask (col_missing_mask) is created to identify missing entries.
If there are missing values, a new column is_nan is created to indicate missing values (1 for missing, 0 for non-missing).

Convert Categorical Columns to Numeric: The convert_to_numeric() function is called to convert categorical columns to numeric values, excluding the column being predicted (col).
Train LightGBM Model:
The rows with non-missing values (train) are used to train the LightGBM model, and the rows with missing values (test) are used for prediction.
The features used for training are all other columns except for the target column (col) and the is_nan column.
The model is trained on train data and used to predict missing values for test.

Impute Missing Values: The predicted values from the model are assigned back to pred_df, replacing the missing values in the original column.
```python
#Create a copy of the Data Frame
pred_df = new_df.copy()
```
```python
predict_missing_values( new_df, column_to_clean, pred_df, False, df_categorical_columns  )
```

1.2.8 Inspect Data frame 
```python
#Displays the first 5 rows of the data frame
pred_df.head()
```
```python
#generates a summary of statistics for the numerical columns in the DataFrame
pred_df.describe()
```
```python
#check for missing (NaN) values in the data frame
new_df.isna().sum().sort_values()
```
From the image above, we have successfully imputed all missing values.2.2.9 Exploratory data analysis: Visualize distributions of the new  data frame

1.2.9.1 Display a side-by-side count plot for each categorical column
```python
figure, axs = plt.subplots( nrows=1, ncols=2, figsize=(20,10))
for i, column in enumerate( df_categorical_columns ):

    sns.countplot( x= pred_df[column],  ax=axs[ i ] )
    axs[ i ].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.9.2 Display histograms for continuous columns
```python
figure, axs = plt.subplots( nrows=3, ncols=2, figsize=(20,20))
for i, column in enumerate( df_continuous_columns ):
    row = i // 2
    col = i % 2
    sns.histplot( pred_df[column],  ax=axs[row, col] )
    axs[row, col].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.9.3 Display boxplots for continuous columns
```python
figure, axs = plt.subplots( nrows=3, ncols=2, figsize=(20,20))
for i, column in enumerate( df_continuous_columns ):
    row = i // 2
    col = i % 2
    sns.boxplot( pred_df[column],  ax=axs[row, col] )
    axs[row, col].set_title(f'{column} data distribution' )
    plt.tight_layout()
plt.show()
```

1.2.9.4 Display pair plots for continuous columns
```python
sns.pairplot( pred_df )
plt.show()
```

1.2.9.5 Identify relationships between variables
```python
sns.heatmap( pred_df[ pred_df.select_dtypes( include= np.number ).columns  ].corr(), annot=True, cmap='Blues' )
plt.show()
```

1.2.10 Encode categorical variable using one hot encoding
```python
final_df = pd.get_dummies( pred_df, columns= df_categorical_columns )
final_df.head()
```
Convert categorical variables into dummy/indicator variables (i.e., one-hot encoding). For each unique category in the column, a new binary column is created, where 1(True) indicates the presence of that category and 0 (False) indicates its absence.

1.2.11 Scale data frame using Robust Scaler
```python
rbs = RobustScaler()
scaled_df = rbs.fit_transform( final_df )
scaled_df = pd.DataFrame( scaled_df, columns=final_df.columns )
scaled_df.head()
```
Robust Scaler is particularly useful when the dataset contains outliers because it scales the data using the median and interquartile range (IQR) rather than the mean and standard deviation. This makes the scaling less sensitive to outliers.

1.2.12 Train the Model
1.2.12.1 Split features
```python
train =  scaled_df.drop('diabetes', axis=1)
test = scaled_df['diabetes']
```
dataset was split into features (independent variables) and target (dependent variable). The features, excluding the diabetes column, were stored in the train set, while the diabetes column, representing the target variable, was stored in the test set.

```python
X_train, X_test, y_train, y_test = train_test_split( train, test, test_size=0.2, random_state=42 )
```
The dataset was further split into training and testing sets using an 80/20 split. The features (independent variables) were assigned to X_train and X_test, while the target variable (diabetes) was assigned to y_train and y_test. The split was performed with a test size of 20%, and the random state was set to 42 to ensure reproducibility of results.

1.2.12.2 Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique
```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("Class distribution before SMOTE:")
print(y_train.value_counts())
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())
```
To address the class imbalance in the training dataset, we applied the Synthetic Minority Over-sampling Technique (SMOTE) with a random seed of 42. SMOTE generated synthetic samples of the minority class to balance the class distribution in the training data. The class distributions before and after applying SMOTE were then compared to assess the impact of the technique.X_test_resampled, y_test_resampled = smote.fit_resample(X_test, y_test)

```python
# Check the class distribution after SMOTE
print("Class distribution before SMOTE:")
print(y_test.value_counts())
print("\nClass distribution after SMOTE:")
print(pd.Series(y_test_resampled).value_counts())
```
To address the class imbalance in the testing dataset, we also  applied the Synthetic Minority Over-sampling Technique (SMOTE) with a random seed of 42. SMOTE generated synthetic samples of the minority class to balance the class distribution in the training data. The class distributions before and after applying SMOTE were then compared to assess the impact of the technique.

1.2.12.3 Initialize and train the Random Forest model
```python
rfc = RandomForestClassifier( class_weight='balanced', random_state=42 )
rfc.fit( X_train_resampled, y_train_resampled )
```
The Random Forest Classifier (RFC) model was initialized with a random state of 42 to ensure reproducibility. The model was then trained on the resampled training data (using SMOTE) to learn the relationship between the features and the target variable.

```python
y_pred = rfc.predict( X_test_resampled )
```
The trained Random Forest model was used to make predictions on the resampled test dataset, and the predicted values were stored in the y_pred variable.

1.2.12.4 Analyse model Performance
```python
acc_score = accuracy_score(y_test_resampled, y_pred)
print(f"{acc_score:.4f}")
```
The accuracy of the Random Forest model on the resampled test dataset was evaluated, yielding an accuracy score of 0.9424(94.24%). This indicates that the model correctly classified approximately 94.24% of the instances in the test setskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

```python
# Initialize model (example: Random Forest)
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Perform Stratified K-Fold Cross-Validation
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(train, test), 1):
    print(f"Fold {fold}")

    # Split data into training and testing sets using .iloc for integer-location based indexing
    X_train, X_test = train.iloc[train_idx], train.iloc[test_idx]
    y_train, y_test = test.iloc[train_idx], test.iloc[test_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_predt = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predt)
    fold_accuracies.append(accuracy)

    print(f"Accuracy for Fold {fold}: {accuracy:.4f}")
```
To evaluate the model's generalization performance, Stratified K-Fold Cross-Validation with 5 splits was performed, ensuring that each fold had the same proportion of class labels as the full dataset. The model used was a Random Forest Classifier with class weights adjusted to handle any class imbalance. For each fold, the model was trained on a subset of the data, and predictions were made on the corresponding test fold. The accuracy for each fold was computed and reported.
The accuracy for each fold is very high, ranging between 96.38% and 96.53% across folds 1 through 5. This suggests the model is performing consistently well on different subsets of the data, which is generally a good sign of a well-generalized model.

```python
class_names = [0, 1]
conf_matrix = confusion_matrix(y_test_resampled, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
True Negatives (TN): 17,888 - the model correctly predicted "0" (negative).
False Positives (FP): 404 - the model incorrectly predicted "1" (positive) for actual "0".
False Negatives (FN): 1,702 - the model incorrectly predicted "0" for actual "1".
True Positives (TP): 16,590 - the model correctly predicted "1" (positive).

```python
class_report = classification_report(y_test_resampled, y_pred, target_names=['0', '1'])
print("\nClassification Report:")
print(class_report)
```
Overall Classification Report:
Precision for class 0 (absence of hypertension): 0.91
Precision for class 1 (presence of hypertension): 0.98
Recall for class 0: 0.98
Recall for class 1: 0.91
F1-score for both classes is 0.94, which represents a good balance between precision and recall.

These metrics indicate that the model is correctly identifying and predicting both classes with a strong balance between precision (correctly identifying positive predictions) and recall (correctly identifying all true positives). This shows that the model is neither overly simplistic (which would lead to underfitting) nor excessively focused on certain patterns at the expense of generalization (which would lead to overfitting).

2.0 Recommendation
The model's performance can be further enhanced by tuning the hyperparameters to optimize the Random Forest classifier's settings. Additionally, addressing the class imbalance by collecting more samples with a target of 1 (positive cases) would help to further balance the dataset. This would allow the model to learn better representations of the minority class, potentially improving its classification accuracy and generalizability

To gain a deeper understanding and enhance your reading experience, feel free to explore more in my Medium article by clicking [here](https://medium.com/@adedokunjuliusayobami/random-forest-diabetes-prediction-1231341c3a79)