from datasets import load_dataset
import pandas as pd

# Load raw text dataset from hugging face andf save it as csv
def load_dataset_raw(sample_size=1500):
    #Load yelp review (full train split)
    dataset = load_dataset('yelp_review_full', split="train")
    #shuffle and select a subset
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    #create a df with only prompt text
    df = pd.DataFrame({"prompt": [x["text"] for x in dataset]})

    #save raw data to CSV
    df.to_csv("data/raw/raw_prompts.csv", index=False)

    return df

if __name__ == "__main__":
    load_dataset_raw()

