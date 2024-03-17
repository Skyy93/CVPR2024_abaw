import pandas as pd
from sklearn.utils import shuffle  # Optional, for shuffling the combined DataFrame

def combine_csv_files(train_csv_path, val_csv_path, combined_csv_path, shuffle_data=True):
    """
    Combine train and validation CSV files into one.

    Parameters:
    - train_csv_path: Path to the training CSV file.
    - val_csv_path: Path to the validation CSV file.
    - combined_csv_path: Path where the combined CSV will be saved.
    - shuffle_data: Whether to shuffle the combined data or not.

    Returns:
    - Path to the combined CSV file.
    """
    # Read the CSV files
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Combine the DataFrames
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # Optional: Shuffle the combined DataFrame
    if shuffle_data:
        combined_df = shuffle(combined_df, random_state=42)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(combined_csv_path, index=False)
    
    return combined_csv_path

# Example usage
combined_csv_path = combine_csv_files('data/train_split.csv', 'data/valid_split.csv', 'data/combined_split.csv')
print(f'Combined CSV saved at: {combined_csv_path}')
