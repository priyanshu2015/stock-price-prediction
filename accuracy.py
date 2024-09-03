import pandas as pd

def calculate_match_percentage(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the DataFrame has at least two columns
    if df.shape[1] < 2:
        raise ValueError("The CSV file must have at least two columns")

    # Extract the last two columns
    last_two_columns = df.iloc[:, -2:]

    # Check for matches
    matches = last_two_columns.iloc[:, 0] == last_two_columns.iloc[:, 1]

    # Calculate the number of matches
    num_matches = matches.sum()

    # Calculate the percentage of matches
    total_rows = len(df)
    match_percentage = (num_matches / total_rows) * 100

    return num_matches, match_percentage

# Example usage
file_path = 'strategy_2_llama-3-sauerkrautlm-70b-instruct.csv'
num_matches, match_percentage = calculate_match_percentage(file_path)
print(f"Number of matches: {num_matches}")
print(f"Percentage of matches: {match_percentage:.2f}%")


# Strategy 1
# Mixtral
# Number of matches: 373
# Percentage of matches: 40.63%