import pandas as pd
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


# ----------------------------- Data Preparation and TF-IDF Implementation -----------------------------

def preprocess_text(text):
    """
    Preprocesses the text by converting it to lowercase, removing punctuation,
    and keeping words longer than 3 letters.

    Parameters:
        text (str): The input text to preprocess.

    Returns:
        list: A list of processed words from the input text.
    """
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return [word for word in text.split() if len(word) > 3]


def preprocess_text_w2v(text):
    """
    Preprocesses the text for Word2Vec training by converting it to lowercase,
    removing punctuation, and keeping words with length >= 3 (to include words like "ron").

    Parameters:
        text (str): The input text to preprocess.

    Returns:
        list: A list of processed words from the input text.
    """
    text = re.sub(r'[^a-z\s]', '', text.lower())
    return [word for word in text.split() if len(word) >= 3]


def load_data(file_path):
    """
    Loads the dataset from the given CSV file and groups dialogues by character.

    Parameters:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        dict: A dictionary where the keys are character names (from 'Character Name')
              and the values are a concatenated string of all dialogues (from 'Dialogue').

    This function addresses subparts 1.1 (Data Preparation) and 1.2.1 (Document Creation).
    """
    df = pd.read_csv(file_path)
    return df.groupby("Character Name")["Dialogue"].apply(
        lambda x: " ".join(str(i) for i in x)).to_dict()


def get_top_words(character_dialogues, characters, top_n=2):
    """
    Identifies the top N most frequent words (with more than 3 letters) in the dialogues of specified characters.

    Parameters:
        character_dialogues (dict): A dictionary mapping character names to their combined dialogues.
        characters (list): A list of character names to consider.
        top_n (int, optional): The number of top words to return for each character. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - A dictionary with each character as key and a list of tuples (word, count) of the top N words.
            - A dictionary mapping each character to a Counter object of all word frequencies.

    This function addresses subpart 1.2.2 (finding the two most frequent words for specific characters).
    """
    character_words = {char: preprocess_text(character_dialogues[char]) for char in characters if
                       char in character_dialogues}
    word_frequencies = {char: Counter(words) for char, words in character_words.items()}
    return {char: freq.most_common(top_n) for char, freq in
            word_frequencies.items()}, word_frequencies


def compute_tf(word, word_freq):
    """
    Computes the Term Frequency (TF) of a given word in a document.

    Parameters:
        word (str): The word for which to compute TF.
        word_freq (Counter): A Counter object representing word frequencies in the document.

    Returns:
        float: The term frequency of the word.
    """
    return word_freq[word] / sum(word_freq.values())


def compute_idf(word, docs):
    """
    Computes the Inverse Document Frequency (IDF) of a given word across multiple documents.

    Parameters:
        word (str): The word for which to compute IDF.
        docs (list): A list of Counter objects representing word frequencies in each document.

    Returns:
        float: The inverse document frequency of the word.
    """
    num_docs_with_word = sum(1 for doc in docs if word in doc)
    return np.log(len(docs) / (1 + num_docs_with_word))


def compute_tf_idf(top_words, word_frequencies):
    """
    Computes the TF-IDF score for the top words for each character.

    Parameters:
        top_words (dict): A dictionary with character names as keys and a list of (word, count) tuples as values.
        word_frequencies (dict): A dictionary mapping character names to their word frequency Counter objects.

    Returns:
        dict: A dictionary mapping each character to another dictionary where keys are words and values are their TF-IDF scores.

    This function addresses subpart 1.2.2 by comparing the TF-IDF representations for the frequent words.
    """
    documents = list(word_frequencies.values())
    return {
        char: {word: compute_tf(word, word_frequencies[char]) * compute_idf(word, documents) for
               word, _ in words}
        for char, words in top_words.items()
    }


# ----------------------------- Word2Vec Model Training and Evaluation -----------------------------

def load_corpus(file_path):
    """
    Loads the CSV file and returns a list of tokenized dialogues for Word2Vec training.

    Parameters:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        list: A list of lists, where each sublist is a tokenized dialogue (list of words) from 'Dialogue'.

    This function helps prepare the corpus for the Word2Vec model training (subpart 1.2.3).
    """
    df = pd.read_csv(file_path)
    sentences = [preprocess_text_w2v(dialogue) for dialogue in df["Dialogue"] if
                 pd.notnull(dialogue)]
    return sentences


def tsnescatterplot(model, word, list_names):
    """
    Creates a 2D t-SNE scatter plot for the central word and a list of related words from the Word2Vec model.

    Parameters:
        model (gensim.models.Word2Vec): The trained Word2Vec model.
        word (str): The central word to use for the plot.
        list_names (list): A list of words to include in the plot along with the central word.

    Returns:
        None: Displays a matplotlib plot.

    This function addresses subpart 1.2.5 (visualization using t-SNE scatter plot).
    """
    arr = np.empty((0, model.vector_size), dtype='f')
    word_labels = []

    # Add the central word if it exists in the vocabulary.
    if word in model.wv:
        word_labels.append(word)
        arr = np.append(arr, [model.wv[word]], axis=0)
    else:
        print(f"'{word}' not in vocabulary.")
        return

    # Add each word from the list if it exists.
    for wrd in list_names:
        if wrd in model.wv:
            word_labels.append(wrd)
            arr = np.append(arr, [model.wv[wrd]], axis=0)

    n_samples = arr.shape[0]
    if n_samples < 2:
        print("Not enough words to create a t-SNE plot.")
        return

    # Set perplexity to a value less than n_samples.
    perplexity = min(30, n_samples - 1)

    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    arr_2d = tsne.fit_transform(arr)
    plt.figure(figsize=(6, 6))
    plt.scatter(arr_2d[:, 0], arr_2d[:, 1])
    for label, x, y in zip(word_labels, arr_2d[:, 0], arr_2d[:, 1]):
        plt.annotate(label, (x, y))
    plt.title("t-SNE Scatter Plot")
    plt.show()


def compare_models(model, model_name):
    """
    Evaluates the Word2Vec model using various similarity and doesn't-match queries.

    Parameters:
        model (gensim.models.Word2Vec): The trained Word2Vec model.
        model_name (str): A label for the model (used in print statements).

    Returns:
        None: Prints the results of various queries.

    This function addresses subpart 1.2.4 (comparing models based on specific queries).
    """
    print(f"\nComparisons for {model_name}:")
    try:
        print("most_similar(positive=['harry']):", model.wv.most_similar(positive=["harry"]))
    except KeyError as e:
        print("harry:", e)
    try:
        print("most_similar(positive=['potter']):", model.wv.most_similar(positive=["potter"]))
    except KeyError as e:
        print("potter:", e)
    try:
        print("most_similar(positive=['hermione']):", model.wv.most_similar(positive=["hermione"]))
    except KeyError as e:
        print("hermione:", e)
    try:
        print("most_similar(positive=['ron']):", model.wv.most_similar(positive=["ron"]))
    except KeyError as e:
        print("ron:", e)
    try:
        print("similarity('spell','charm'):", model.wv.similarity('spell', 'charm'))
    except KeyError as e:
        print("spell/charm:", e)
    try:
        print("similarity('gryffindor','slytherin'):",
              model.wv.similarity('gryffindor', 'slytherin'))
    except KeyError as e:
        print("gryffindor/slytherin:", e)
    try:
        print("doesnt_match(['potion', 'wand', 'arithmetic']):",
              model.wv.doesnt_match(['potion', 'wand', 'arithmetic']))
    except KeyError as e:
        print("potion/wand/arithmetic:", e)
    try:
        print("doesnt_match(['harry', 'draco', 'hermione']):",
              model.wv.doesnt_match(['harry', 'draco', 'hermione']))
    except KeyError as e:
        print("harry/draco/hermione:", e)
    try:
        print("doesnt_match(['ron', 'crookshanks', 'hedwig']):",
              model.wv.doesnt_match(['ron', 'crookshanks', 'hedwig']))
    except KeyError as e:
        print("ron/crookshanks/hedwig:", e)
    try:
        print("most_similar(positive=['magic','hogwarts'], negative=['school'], topn=3):",
              model.wv.most_similar(positive=["magic", "hogwarts"], negative=["school"], topn=3))
    except KeyError as e:
        print("magic/hogwarts/school:", e)
    try:
        print("most_similar(positive=['curse','ron'], negative=['harry'], topn=3):",
              model.wv.most_similar(positive=["curse", "ron"], negative=["harry"], topn=3))
    except KeyError as e:
        print("curse/ron/harry:", e)


# -------------------------------------- Main Execution Block --------------------------------------

if __name__ == "__main__":
    file_path = "dataset/harry_potter_characters_with_dialogues.csv"

    # --------------------- 1.1 Data Preparation and 1.2.1 Document Creation ---------------------
    # Load dataset and group dialogues by character using 'Character Name' and 'Dialogue'.
    character_dialogues = load_data(file_path)

    # --------------------- 1.2.2 TF-IDF Representation ---------------------
    # Select characters of interest and extract the top 2 words from each (words > 3 letters).
    selected_characters = ["Harry Potter", "Ron Weasley", "Hermione Granger"]
    top_words, word_frequencies = get_top_words(character_dialogues, selected_characters)
    tf_idf_scores = compute_tf_idf(top_words, word_frequencies)
    print("Top Words for Each Character:", top_words)
    print("\nTF-IDF Scores:", tf_idf_scores)

    # --------------------- 1.2.3 Word2Vec Model Training ---------------------
    # Load and prepare the corpus for Word2Vec training (each dialogue is tokenized with preprocess_text_w2v).
    sentences = load_corpus(file_path)
    print(f"\nNumber of sentences for Word2Vec training: {len(sentences)}")

    # Train the first Word2Vec model with: context window = 5, vector dimensions = 100, negative samples = 5 (skip-gram).
    print(
        "\nTraining Word2Vec model with context window=5, vector dimensions=100, negative samples=5")
    model1 = Word2Vec(sentences=sentences, vector_size=100, window=5, negative=5, sg=1, min_count=1,
                      workers=4)

    # Train the second Word2Vec model with: context window = 10, vector dimensions = 300, negative samples = 15 (skip-gram).
    print(
        "\nTraining Word2Vec model with context window=10, vector dimensions=300, negative samples=15")
    model2 = Word2Vec(sentences=sentences, vector_size=300, window=10, negative=15, sg=1,
                      min_count=1, workers=4)

    # --------------------- 1.2.4 Model Comparison ---------------------
    # Evaluate both models using various similarity and doesnt_match queries.
    print("\nEvaluating Model 1:")
    compare_models(model1, "Model 1")
    print("\nEvaluating Model 2:")
    compare_models(model2, "Model 2")

    # --------------------- 1.2.5 Visualization ---------------------
    # Create t-SNE scatter plots for the neighborhood of "harry" with a list of related words.
    words_to_plot = ['draco', 'ron', 'hermione', 'curse', 'magic', 'dark', 'wizard', 'quidditch']
    print("\nGenerating t-SNE scatter plot for Model 1 (centered on 'harry'):")
    tsnescatterplot(model1, 'harry', words_to_plot)
    print("\nGenerating t-SNE scatter plot for Model 2 (centered on 'harry'):")
    tsnescatterplot(model2, 'harry', words_to_plot)
