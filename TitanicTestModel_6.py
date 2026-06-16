import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#---------------------------------------------------------------
# Function name : PreserveModel
# Description : It displays the formatted title
# Parameters : title (str)
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def LoadPreservedModel(filename):
    loaded_model = joblib.load(filename)
    
    print("\nModel successfully loaded")
    
    return loaded_model

#---------------------------------------------------------------
# Function name : PreserveModel
# Description : It displays the formatted title
# Parameters : title (str)
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def PreserveModel(model, filename):
    joblib.dump(model, filename)
    
    print("\nModel preserved successfully with name : ", filename)
    
#---------------------------------------------------------------
# Function name : TrainTitanicModel
# Description : It displays the formatted title
# Parameters : title (str)
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def TrainTitanicModel(df):
    # Split features and labels
    X = df.drop("Survived"), axis = 1
    Y = df["Survived"]
    
    print("\nFeatures : ")
    print(X.head())
    
    print("\nLabels : ")
    print(Y.head())
    
    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("Y_train shape : ", Y_train.shape)
    print("Y_test shape : ", Y_test.shape)
    
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, Y_train)
    
    print("\nModel trained successfully")
    
    print("\nIntercept of model : ")
    print(model.intercept_)
    
    print("\nCoefficient of model : ")
    for feature, coefficient in zip(X.columns, model.coef_[0]):
        print(feature, ":", coefficient)
        
    PreserveModel(model, "marvelloustitanic.pkl")
    
    loaded_model = LoadPreservedModel("marvelloustitanic.pkl")
    
    Y_pred = loaded_model.predict(X_test)
    
    accuracy = accuracy_score(Y_pred, Y_test)
    
    print("\nAccuracy is : ", accuracy)
    
    cm = confusion_matrix(Y_pred, Y_test)
    
    print("\nConfusion matrix is : ", cm)
    
#---------------------------------------------------------------
# Function name : DisplayInfo
# Description : It displays the formatted title
# Parameters : title (str)
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def DisplayInfo(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

#---------------------------------------------------------------
# Function name : ShowData
# Description : It shows the basic information of the dataset
# Parameters : df
#              df -> pandas dataframe object
#              message
#              message -> Heading text to display  
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def ShowData(df, message):
    DisplayInfo(message)
    
    print("\nFirst 5 rows of the dataset : ")
    print(df.head())
    
    print("\nShape of the dataset : ")
    print(df.shape)
    
    print("\nColumn names in the dataset : ")
    print(df.columns.tolist())
    
    print("\nMissing values in each column : ")
    print(df.isnull().sum())
    
#---------------------------------------------------------------
# Function name : CleanTitanicData
# Description : It does preprocessing
#               It romoves unnecessary columns 
#               It handles missing values
#               It converts text data to numeric format
#               It does encoding of categorical columns
# Parameters : df -> pandas dataframe object
# Return value : df -> Cleaned pandas dataframe object
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original data")
    print(df.head())
    
    # Remove unnecessary columns
    drop_columns = ["Passengerid", "zero", "Name", "Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]
    
    print("\nColumns to be dropped : ")
    print(existing_columns)
    
    # drop the unwanted columns
    df = df.drop(columns = existing_columns)
    DisplayInfo("Step 2 : Data after columns removal")
    print(df.head())
    
    # Handle age column
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # coerce -> Invalid value gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        
        age_median = df["Age"].median()
        print("\nMedian of Age column is : ", age_median)
        
        # Replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)
        
        print("\nAge column after preprocessing : ")
        print(df["Age"].head(10))
        
    # Handle fare column
    if "Fare" in df.columns:
        print("\nFare column before preprocessing")
        print(df["Fare"].head(10))
        
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        
        fare_median = df["Fare"].median()
        print("\nMedian of Fare column is : ", fare_median)
        
        # Replace missing values with median
        df["Fare"] = df["Fare"].fillna(age_median)
        
        print("\nFare column after preprocessing : ")
        print(df["Fare"].head(10))
      
    # Handle Embarked column
    if "Embarked" in df.columns:
        print("\nEmbarked column before preprocessing")
        print(df["Embarked"].head(10))
        
        # Convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()
        
        # Remove missing values
        df["Embarked"] = df["Embarked"].replace(['nan', 'None', ''], np.nan)
        
        # Get most frequent value
        embarked_mode = df["Embarked"].mode()[0]
        print("\nMode of embarked column : ", embarked_mode)
        
        # Replace missing values with mode
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        
        print("\nEmbarked column after preprocessing : ")
        print(df["Embarked"].head(10))
        
    # Handle Sex column
    if "Sex" in df.columns:
        print("\nSex column before preprocessing")
        print(df["Sex"].head(10))
        
        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")
        
        print("\nSex column after preprocessing : ")
        print(df["Sex"].head(10))
        
    DisplayInfo("Data after preprocessing")
    print(df.head())
    
    print("\nMissing values after preprocessing")
    print(df.isnull().sum())
    
    # Encode Embarked column
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    print("\nData after encoding")
    
    print(df.head())
    
    print("\nShape of dataset : ", df.shape)
    
    # Convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    print("\nData after encoding")
    
    print(df.head())
    
    return df

#---------------------------------------------------------------
# Function name : MarvellousTitanicLogistic
# Description : It is the main pipeline conntroller
#               It loads the dataset and displays raw data
#               It preprocess the dataset & train the model 
# Parameters : Datapath (str) : Path to the dataset file
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def MarvellousTitanicLogistic(Datapath):
    DisplayInfo("Step 1 : Loading the dataset")
    
    df = pd.read_csv(Datapath)
    
    ShowData(df, "Initial Dataset Information")
    
    df = CleanTitanicData(df)
    
    TrainTitanicModel(df)
    
#---------------------------------------------------------------
# Function name : main
# Description : Starting point of the application
# Parameters : None
# Return value : None
# Date : 14/03/2026
# Author : Aparna Tukaram Shinde
#---------------------------------------------------------------

def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")
    
if __name__ == "__main__":
    main()