import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Define file path
file_path = r"C:\Users\niran\OneDrive\Dokumen\cus-acore.csv"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Load dataset
    df = pd.read_csv(file_path)

    # Print column names to check
    print(df.columns)

    # Adjusted feature selection based on correct column names
    X = df[['AnnualIncome', 'SpendingScore']]

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X)

    # Function to classify new input data
    def classify_customer(annual_income, spending_score):
        input_data = np.array([[annual_income, spending_score]])
        cluster = kmeans.predict(input_data)
        return cluster[0]

    # Plotting the clusters
    plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segments')
    plt.show()

    # Displaying the dataframe with cluster information
    print(df)

    # Prompt user for input
    annual_income = float(input("Enter the annual income (in k$): "))
    spending_score = float(input("Enter the spending score (1-100): "))

    # Classify user input
    cluster = classify_customer(annual_income, spending_score)
    print(f"The customer with Annual Income {annual_income}k$ and Spending Score {spending_score} belongs to cluster {cluster}.")
