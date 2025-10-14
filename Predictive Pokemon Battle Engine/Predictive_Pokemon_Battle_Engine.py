#IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('Pokemon.csv')     #loads dataframe into project

#DATA CLEANING
df = df.rename(columns={'#':'ID'})      #rename the '#' column to'ID' for easier access

df['Type 2'] = df['Type 2'].fillna('None')      #df['Type 2'] isolates and returns a list of the 'Type 2' column, fillna('None') replaces all empty vales with 'None'

df = df[df['Generation'] == 1]     #keep only the original generation of pokemon FOR NOW (reassign df to gen 1)


"""This was me just playing around with these two libraries to get a grasp of what they did
Code isnt necessary to keep just wanted to showcase to anyone looking at my code"""

"""Plot 1: Attack and Defense Stats Comparison

plt.figure(figsize=(10, 6))     #determines the size of the window
sns.scatterplot(x='Attack', y='Defense', data=df, hue='Legendary')      #draws a scatter graph 
plt.title('Attack and Defense Stats Comparison')
plt.grid(True)
plt.show()"""

"""Plot 2: Attack and Speed Stats Comparison

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Attack', y='Speed', data=df, hue='Legendary')
plt.title('Attack and Speed Stats Comparison')
plt.grid(True)
plt.show()"""

"""Plot 3: How many Pokemon of each type as their primary typing?

plt.figure(figsize=(12, 7))     #determines the size of the window
sns.countplot(data=df, y='Type 1', order=df['Type 1'].value_counts().index)     #draws a bargraph and orders it from most to least common type from top to bottom
plt.title('Number of Pokemon per Primary Type')     #Heads the graph
plt.grid(True)      #shows a grid pattern on the graph
plt.show()      #actually displays the window with all the previous commands on"""

#FEATURE ENGINEERING

"""Goal of this section is to simulate every pokemon fighting every other pokemon within the dataframe"""

stats_df = df[['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

#Create two copies of the data frame each with a different suffix to the column name to prevent column name clashes when merging the dataframes
df1 = stats_df.copy().add_suffix('_1')
df2 = stats_df.copy().add_suffix('_2')

#Here I am creating a new column within the two df's to then merge them later using that key
df1['key'] = 1
df2['key'] = 1

battle_df = pd.merge(df1, df2, on='key').drop('key', axis=1)    #merges the two df's using the key column but then removes the key column after

battle_df = battle_df[battle_df['Name_1'] != battle_df['Name_2']]   #Removes matchups where a pokemon would fight itself

""" Checking mathematically that battle_df has the right elements:

        print(df) says there are 166 entries, 

        this means that if every pokemon fought every other but not their own species the calculation would be 166 * 165 = 27390,

        print(battle_df) says there are 27390 entries which confirms mathematical belief
"""

#Determining the Winner
def calculate_winner(row):

    score_1 = row[['HP_1', 'Attack_1', 'Defense_1', 'Sp. Atk_1', 'Sp. Def_1']].sum()
    score_2 = row[['HP_2', 'Attack_2', 'Defense_2', 'Sp. Atk_2', 'Sp. Def_2']].sum()

    if score_1 > score_2:
        return 1    #Pokemon 1 wins
    elif score_2 > score_1:
        return 0    #Pokemon 2 wins
    else:
        return 1 if row['Speed_1'] > row['Speed_2'] else 0 # Speed is the tie-breaker

battle_df['Winner'] = battle_df.apply(calculate_winner, axis=1)     #applying the method to every row to create target 'Winner'

#Creating numerical dataframe of difference in stats
feature_df = pd.DataFrame()
feature_df['HP_diff'] = battle_df['HP_1'] - battle_df['HP_2']
feature_df['Attack_diff'] = battle_df['Attack_1'] - battle_df['Attack_2']
feature_df['Defense_diff'] = battle_df['Defense_1'] - battle_df['Defense_2']
feature_df['Sp_Atk_diff'] = battle_df['Sp. Atk_1'] - battle_df['Sp. Atk_2']
feature_df['Sp_Def_diff'] = battle_df['Sp. Def_1'] - battle_df['Sp. Def_2']
feature_df['Speed_diff'] = battle_df['Speed_1'] - battle_df['Speed_2']

#MODEL BUILDING AND TRAINING (using Scikit-learn)

#Defining an X (using the feature matrix) and a target vector y (winner)
X = feature_df
y = battle_df['Winner']

#Splitting X and y in a 4:1 split of train:test of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create an instance of the model
model = LogisticRegression(max_iter=1000)   #helps with convergence

#Train the model
model.fit(X_train, y_train)

#TESTING THE MODEL

y_pred = model.predict(X_test)

#Calculating accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix as a heatmap
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['P2 Wins', 'P1 Wins'], yticklabels=['P2 Wins', 'P1 Wins'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()