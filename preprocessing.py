import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import DATA_PATH

nltk.download('punkt')
nltk.download('stopwords')

def load_data(path=DATA_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item["question"].lower(), item["answer"].lower()) for item in data]

def tokenize(sentence):
    stop_words = set(stopwords.words('english'))
    return [word for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]

def build_vocab(pairs):
    tokens = set(word for pair in pairs for sent in pair for word in tokenize(sent))
    word2idx = {word: idx+2 for idx, word in enumerate(sorted(tokens))}
    word2idx["<PAD>"] = 0
    word2idx["<SOS>"] = 1
    word2idx["<EOS>"] = len(word2idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word
