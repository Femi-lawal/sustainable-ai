#Compute NLP and derived numeric features for prompts

from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords

#Downlaod stopwrods if not done already

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words('english'))

#Load DistilBERT tokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def compute_feature(prompt, num_layers, training_hours, flops_per_hour):
    #Compute features for a given prompt and model params



    word = prompt.split()
    chars = len(prompt)

    #token count using tokenizer
    token_counter = len(tokenizer.encode(prompt))

    #ratio of punctiuatioin characters
    punct_ratio = sum(1 for c in prompt if c in ".,!?;:") /max(chars,1)

    #Average word length
    avg_word_len = sum(len(w) for w in word) /max(len(word),1)


    #Ratio of stopwrods

    stopword_ratio = sum(1 for w in word if w.lower() in stop_words) /max(len(word),1)

    #Derived numeric features
    flops_per_layer = flops_per_hour / max(num_layers,1)
    training_efficiency = training_hours / max(num_layers,1)

    return {
        "prompt": prompt,
        "token_count": token_counter,
        "char_count": chars,
        "punct_ratio": punct_ratio,
        "avg_word_length": avg_word_len,
        "stopword_ratio": stopword_ratio,
        "num_layers": num_layers,
        "training_hours": training_hours,
        "flops_per_hour": flops_per_hour,
        "flops_per_layer": flops_per_layer,
        "training_efficiency": training_efficiency,

    }