import pandas as pd

# Load the dataset (you already have this)
df = pd.read_csv(r"C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\reward_dataset.csv")

# Split input_text into query and answer based on " [SEP] "
df[['query', 'answer']] = df['input_text'].str.split(r' \[SEP\] ', expand=True)

# Separate good and bad feedback entries
good_df = df[df['label'] == 1]
bad_df = df[df['label'] == 0]

# Merge on 'query' to pair good and bad answers
pairs_df = pd.merge(
    good_df[['query', 'answer']],
    bad_df[['query', 'answer']],
    on='query',
    suffixes=('_preferred', '_dispreferred')
)

# Optional: Remove duplicates
pairs_df = pairs_df.drop_duplicates()

# Save to CSV for DPO training input
pairs_df.to_csv(r"C:\Users\sneha\Downloads\projectdsmm\chatter_files\Project\CodesToUpload\rlhf\preference_pairs.csv", index=False)

print(f"Preference pairs prepared: {len(pairs_df)}")
