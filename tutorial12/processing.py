import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

DIR = 'tutorial12\dataset'
def ProcessData():
    #########################Reading#########################
    file_path = os.path.join(os.getcwd(), DIR)
    data_df = pd.read_csv(os.path.join(file_path, os.listdir(file_path)[0]))
    print(data_df.head())
    #########################Cleaning#########################
    data_df['Genre'] = data_df['Genre'].map({'Male': 0, 'Female': 1})
    data_df.drop(['CustomerID'], axis=1, inplace=True)
    data_df.fillna(data_df.mean(axis=0), inplace=True)
    print(data_df.head())
    X = data_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    #########################Scaling#########################
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    ##########################DImensionality Reduction#########################
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    print(X)
    return X
    
if __name__ == "__main__":
    ProcessData()