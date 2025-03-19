import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




filename = "data.csv"

#download file from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/download
df = pd.read_csv(filename)

print(df.head(5))

print(df.tail(5))

print(df.info())

print(df.describe())

#check for missing values
df.isna().sum()  

#drop columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True) #1 = column, 0 = row
print(df.head(5))

#scaling the data
df_copy = df.copy() #just to preserve the original dataframe

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaler= scaler.fit_transform(df.iloc[:, 1:]) #returns a numpy array
df_scaled = pd.DataFrame(df_scaler, columns=df.columns[1:]) #converts back to pandas dataframe
print(df_scaled.head(5))

#Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df.iloc[:, 0] = encoder.fit_transform(df.iloc[:, 0])
print(df.head(5))

#visualizations
df_without_first_column = df.iloc[:, 1:]
correlation_matrix = df_without_first_column.corr()
print(correlation_matrix)

#correlation_matrix heat map
import seaborn as sns
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt= ".2f", linewidth=0.5)
plt.show()

#box plot
plt.figure(figsize=(20, 15))
sns.boxplot(x="diagnosis", y="radius_mean", data=df)
plt.show()

#violin plot
plt.figure(figsize=(10,5))
sns.violinplot(x="diagnosis", y="area_se", data=df, palette='muted')
plt.show()