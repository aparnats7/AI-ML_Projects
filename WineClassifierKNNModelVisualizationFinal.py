import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier(DataPath):
    border = "-" * 50
    
    # Step 1: Load the dataset from the CSV file
    print(border)
    print("Step 1: Load the dataset from the CSV file")
    print(border)
    
    df = pd.read_csv(DataPath)
    print("Some entries from the dataset :")
    print(df.head())
    print(border)
    
    # Step 2: Clean the dataset by removing empty rows
    print(border)
    print("Step 2: Clean the dataset by removing empty rows")
    print(border)
    
    df.dropna(inplace=True)
    print("Total records : ", df.shape[0])
    print("Total columns : ", df.shape[1])
    print(border)
    
    # Step 3: Separate independent and dependent variables
    print(border)
    print("Step 3: Separate independent and dependent variables")
    print(border)
    
    X = df.drop(columns=['Class'])
    Y = df['Class']
    
    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)
    
    print(border)
    print("Input columns : ", X.columns.tolist())
    print("Output column : Class ")
    
    # Step 4: Split the dataset into training and testing sets
    print(border)
    print("Step 4: Split the dataset into training and testing sets")
    print(border)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(border)
    print("Information of training and testing data :")
    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("Y_train shape : ", Y_train.shape)
    print("Y_test shape : ", Y_test.shape)
    print(border)
    
    # Step 5: Feature Scaling
    print(border)
    print("Step 5: Feature Scaling")
    print(border)
    
    scaler = StandardScaler()
    # Independent variables scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    print("Feature scaling is done successfully")
    
    # Step 6: Explore multiple values of K
    # Hyperparameter tuning (K)
    print(border)
    print("Step 6: Explore multiple values of K")
    print(border)
    
    accuracy_scores = []
    k_values = range(1, 21)
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)
        
    print(border)
    print("Accuracy report of all K values from 1 to 20 :")
    for value in accuracy_scores:
        print(value)
        
    print(border)

    # Step 7: Plot graph of K values vs Accuracy scores
    print(border)
    print("Step 7: Plot graph of K values vs Accuracy scores")
    print(border)
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_scores, marker = 'o')
    plt.title('K values vs Accuracy scores')
    plt.xlabel('K values')
    plt.ylabel('Accuracy Scores')
    plt.grid(True)
    plt.xticks(list(k_values))
    plt.show()

    # Step 8: Find the best K value
    print(border)
    print("Step 8: Find the best K value")
    print(border)
    
    best_k = list(k_values)[accuracy_scores.index(max(accuracy_scores))]
    
    print("Best value of K : ", best_k)
    
    # Step 9: Build the final KNN model using the best K value
    print(border)
    print("Step 9: Build the final model using the best K value")
    print(border)
    
    final_model = KNeighborsClassifier(n_neighbors = best_k)
    final_model.fit(X_train_scaled, Y_train)
    Y_pred = final_model.predict(X_test_scaled)
    
    # Step 10: Calculate final accuracy
    print(border)
    print("Step 10: Calculate final accuracy")
    print(border)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy of model is : ", accuracy * 100)
    
    # Step 11: Display confusion matrix
    print(border)
    print("Step 11: Display confusion matrix")
    print(border)
    
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix : ")
    print(cm)

    # Step 12: Display classification report
    print(border)
    print("Step 12: Display classification report")
    print(border)
    
    cr = classification_report(Y_test, Y_pred)
    print("Classification Report : ")
    print(cr)
    
def main():
    border = "-" * 50
    print(border)
    print("Wine Classifier using KNN")
    print(border)
    
    MarvellousClassifier("WinePredictor.csv")
    
if __name__ == "__main__":
    main()