import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('preds.csv')

# Columns to check for negative values
columns_to_check = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']

# Set negative values to 0.0 in the specified columns
for column in columns_to_check:
    df[column] = df[column].apply(lambda x: max(x, 0.0)) 
    df[column] = df[column].apply(lambda x: min(x, 1.0)) 
# Save the modified DataFrame back to 'pred.csv'
df.to_csv('preds_modified.csv', index=False)
