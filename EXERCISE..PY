import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

def tokenize(text):
    tokens = word_tokenize(text.lower())
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def pos_tagging(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens

def count_pos(tagged_tokens):
    pos_counts = {}
    for _, pos in tagged_tokens:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pos_counts

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

def plot_pos_frequency(pos_counts):
    pos_freq = FreqDist(pos_counts)
    pos_freq.plot(5, cumulative=False)

def main():
    moby_dick = gutenberg.raw("melville-moby_dick.txt")

    tokens = tokenize(moby_dick)

    filtered_tokens = remove_stopwords(tokens)

    tagged_tokens = pos_tagging(filtered_tokens)

    pos_counts = count_pos(tagged_tokens)

    top_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 POS and their frequencies:")
    for pos, count in top_pos:
        print(f"{pos}: {count}")

    lemmas = lemmatize(tokens[:20])
    print("\nLemmatized tokens:")
    print(lemmas)

    plot_pos_frequency(tagged_tokens)

if __name__ == "__main__":
    main()