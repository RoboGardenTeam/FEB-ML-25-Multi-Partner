import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "tutorial7/results.csv"
df = pd.read_csv(file_path)

# 1. Count of tuples (rows) in the dataset
print("Total matches:", len(df))

# 2. Count of unique tournament names
print("Unique tournaments:", df['tournament'].nunique())

# 3. Count of matches in 2018
print("Matches in 2018:", df[df['date'].str.contains("2018")].shape[0])

# 4. Count of wins, losses, and draws
win_count = (df['home_score'] > df['away_score']).sum()
loss_count = (df['home_score'] < df['away_score']).sum()
draw_count = (df['home_score'] == df['away_score']).sum()
print(f"Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")

# 5. Pie chart of wins, losses, and draws
plt.figure(figsize=(6, 6))
plt.pie([win_count, loss_count, draw_count], labels=['Wins', 'Losses', 'Draws'], autopct='%1.1f%%', colors=['green', 'red', 'blue'])
plt.title("Match Outcomes")
plt.show()

# 6. Pie chart of the neutral column
df['neutral'].value_counts().plot(kind='pie', labels=['Not Neutral', 'Neutral'], autopct='%1.1f%%', colors=['purple', 'orange'])
plt.title("Neutral Venue Matches")
plt.ylabel("")
plt.show()

# 7. Count of unique team names
teams = pd.concat([df['home_team'], df['away_team']]).nunique()
print("Unique teams:", teams)