"""Fix data files to have proper columns."""
import pandas as pd
import numpy as np

np.random.seed(42)

# Load raw prompts to get categories
raw_df = pd.read_csv('data/raw/raw_prompts.csv')

# Create energy dataset with proper columns
prompt_ids = raw_df['prompt_id'].tolist()

# Map categories to energy ranges
categories = {
    'simple': (0.5, 1.5),
    'question': (1.0, 2.5),
    'explanation': (2.0, 4.0),
    'comparison': (2.5, 4.5),
    'creative': (3.0, 5.0),
    'coding': (3.5, 6.0),
    'complex': (5.0, 8.0),
}

# Category assignments from raw_prompts.csv
category_assignments = raw_df['category'].tolist()

energy_data = []
for i, pid in enumerate(prompt_ids):
    cat = category_assignments[i]
    low, high = categories[cat]
    energy = np.random.uniform(low, high)
    
    if energy < 3:
        label = 'efficient'
    elif energy < 5:
        label = 'moderate'
    else:
        label = 'inefficient'
    
    energy_data.append({
        'prompt_id': pid,
        'energy_joules': round(energy, 4),
        'efficiency_label': label
    })

df = pd.DataFrame(energy_data)
df.to_csv('data/synthetic/energy_dataset.csv', index=False)
print(f"Created energy_dataset.csv with {len(df)} rows")
print(df.head(10))
print("\nLabel distribution:")
print(df['efficiency_label'].value_counts())

# Also create features.csv with matching prompt_ids
features_data = []
for i, row in raw_df.iterrows():
    prompt = row['prompt']
    words = prompt.split()
    features_data.append({
        'prompt_id': row['prompt_id'],
        'token_count': len(words) + np.random.randint(0, 5),
        'word_count': len(words),
        'char_count': len(prompt),
        'avg_word_length': round(np.mean([len(w) for w in words]), 2) if words else 0,
        'complexity_score': round(len(prompt) / 20 + np.random.random(), 2),
        'question_mark': 1 if '?' in prompt else 0,
        'exclamation_mark': 1 if '!' in prompt else 0,
    })

features_df = pd.DataFrame(features_data)
features_df.to_csv('data/processed/features.csv', index=False)
print(f"\nCreated features.csv with {len(features_df)} rows")
