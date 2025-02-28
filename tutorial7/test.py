import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "tutorial7/results.csv"
df = pd.read_csv(file_path)
win_count = (df['home_score'] > df['away_score']).sum()
loss_count = (df['home_score'] < df['away_score']).sum()
draw_count = (df['home_score'] == df['away_score']).sum()

plt.figure(figsize=(6, 6))
plt.pie([win_count, loss_count, draw_count], labels=['Wins', 'Losses', 'Draws'], autopct='%1.1f%%', colors=['green', 'red', 'blue'])
plt.title("Match Outcomes")
plt.show()

df['neutral'].value_counts().plot(kind='pie', labels=['Not Neutral', 'Neutral'], autopct='%1.1f%%', colors=['purple', 'orange'])
plt.title("Neutral Venue Matches")
plt.ylabel("")
plt.show()