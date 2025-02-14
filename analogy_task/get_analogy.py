"""
Part II: Word Embeddings Analogy Task and Evaluation

This script performs the following:
	1.	Loads two pretrained embedding models.
	2.	Parses an analogy questions file (word-test.v1.txt) retaining only the allowed eight relation groups.
	3.	Implements the analogy prediction method (using vector arithmetic as described in Equation 2).
	4.	Evaluates the models on the analogy test and prints accuracy results.
	5.	Retrieves the top 10 similar words for selected verbs (to illustrate the antonym issue).
	6.	(Optional) Runs two sets of new analogy tests.
"""

import gensim.downloader as api


def load_pretrained_models():
    """
    Loads two different pretrained word embedding models for comparison.
    Returns:
        tuple: A pair of gensim models:
            model1 (gensim.models.keyedvectors.KeyedVectors): word2vec-google-news-300.
            model2 (gensim.models.keyedvectors.KeyedVectors): glove-wiki-gigaword-100.
    """
    print("Loading Model 1: word2vec-google-news-300 ...")
    model1 = api.load("word2vec-google-news-300")
    print("Loading Model 2: glove-wiki-gigaword-100 ...")
    model2 = api.load("glove-wiki-gigaword-100")
    return model1, model2


def parse_analogy_file(filepath, allowed_groups):
    """
    Parses an analogy questions file and returns only the questions
    belonging to the allowed relation groups.
    Each analogy question should have four words: a, b, c, d,
    representing "a is to b as c is to d".

    Parameters:
        filepath (str): Path to the analogy questions file.
        allowed_groups (set): Set of allowed relation group names (e.g., "capital-world").

    Returns:
        list: A list of tuples (a, b, c, d, group) for each analogy question.
    """
    analogies = []
    current_group = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                # New group; remove colon and any extra spaces.
                current_group = line[1:].strip()
            else:
                # Include question only if its group is allowed.
                if current_group in allowed_groups:
                    words = line.split()
                    if len(words) >= 4:
                        a, b, c, d = words[:4]
                        analogies.append((a, b, c, d, current_group))
    return analogies


def predict_analogy(model, a, b, c):
    """
    Predicts the word d for the analogy “a is to b as c is to d” using vector arithmetic.
    The operation performed is:
          result_vector = v(a) - v(b) + v(c)
    The function then returns the word (not among a, b, or c) with the highest cosine similarity.

    Parameters:
        model: gensim pretrained embedding model.
        a (str): First word.
        b (str): Second word.
        c (str): Third word.

    Returns:
        str or None: The predicted word d, or None if any word is missing in the vocabulary.
    """
    try:
        candidates = model.most_similar(positive=[c, a], negative=[b], topn=10)
    except KeyError:
        return None

    for word, score in candidates:
        if word not in {a, b, c}:
            return word
    return None


def evaluate_analogy_test(model, analogies):
    """
    Evaluates the model’s performance on an analogy test.
    For each analogy (a, b, c, d, group), the function predicts a word and
    compares it (case-insensitively) to the expected d.

    Parameters:
        model: gensim pretrained embedding model.
        analogies (list): List of analogy tuples (a, b, c, d, group).

    Returns:
        tuple: (accuracy, total_questions, correct_predictions)
    """
    total = 0
    correct = 0
    for a, b, c, d, group in analogies:
        prediction = predict_analogy(model, a, b, c)
        if prediction is not None:
            total += 1
            if prediction.lower() == d.lower():
                correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, total, correct


def show_similar_words(model, word, topn=10):
    """
    Retrieves the topn most similar words to a given word.
    Parameters:
        model: gensim pretrained embedding model.
        word (str): The target word.
        topn (int): Number of similar words to retrieve (default is 10).

    Returns:
        list or None: List of tuples (word, similarity score) or None if the word is absent.
    """
    try:
        similar = model.most_similar(positive=[word], topn=topn)
        return similar
    except KeyError:
        return None


def run_optional_new_tests(model):
    """
    Runs optional new analogy test questions that are not part of the original dataset.
    Two sets of new test questions are defined. Each test question is a tuple:
         (a, b, c, expected_d)

    Parameters:
        model: gensim pretrained embedding model.

    Returns:
        tuple: Two lists containing the test results for each set.
               Each result is a tuple (a, b, c, expected_d, predicted_d).
    """

    new_tests_set1 = [
        ("chef", "knife", "painter", "brush"),
        ("driver", "car", "pilot", "plane"),
        ("writer", "pen", "artist", "brush")
    ]
    new_tests_set2 = [
        ("computer", "keyboard", "phone", "touchscreen"),
        ("teacher", "classroom", "doctor", "hospital"),
        ("singer", "microphone", "actor", "camera")
    ]
    results_set1 = []
    results_set2 = []

    for a, b, c, expected in new_tests_set1:
        pred = predict_analogy(model, a, b, c)
        results_set1.append((a, b, c, expected, pred))

    for a, b, c, expected in new_tests_set2:
        pred = predict_analogy(model, a, b, c)
        results_set2.append((a, b, c, expected, pred))

    return results_set1, results_set2


def main():
    """
    Main function to run the analogy evaluation and similar word retrieval.
    The function performs the following steps:
  1. Loads two pretrained models.
  2. Parses the analogy file (word-test.v1.txt) filtering for allowed groups.
  3. Evaluates the models on the analogy test and prints the accuracy.
  4. Retrieves and prints the top 10 similar words for the verbs "increase" and "enter".
  5. Runs optional new analogy tests and displays the results.
    """
    # Load the two pretrained embedding models.
    model1, model2 = load_pretrained_models()

    # Define allowed analogy groups.
    allowed_groups = {
        "capital-world", "currency", "city-in-state", "family",
        "gram1-adjective-to-adverb", "gram2-opposite", "gram3-comparative",
        "gram6-nationality-adjective"
    }

    # Parse the analogy questions file.
    analogy_filepath = "dataset/word-test.v1.txt"
    analogies = parse_analogy_file(analogy_filepath, allowed_groups)
    print(f"\nLoaded {len(analogies)} analogy questions from allowed groups.")

    # Evaluate the analogy test for both models.
    acc1, total1, correct1 = evaluate_analogy_test(model1, analogies)
    acc2, total2, correct2 = evaluate_analogy_test(model2, analogies)

    print("\nAnalogy Test Results:")
    print(
        f"Model 1 (glove-wiki-gigaword-100): Accuracy = {acc1 * 100:.2f}% ({correct1} correct out of {total1})")
    print(
        f"Model 2 (glove-twitter-200): Accuracy = {acc2 * 100:.2f}% ({correct2} correct out of {total2})")

    # Retrieve and display similar words for selected verbs.
    for test_word in ["increase", "enter"]:
        sim1 = show_similar_words(model1, test_word)
        sim2 = show_similar_words(model2, test_word)
        print(f"\nTop 10 words similar to '{test_word}' in Model 1:")
        if sim1:
            for word, score in sim1:
                print(f"  {word}: {score:.4f}")
        else:
            print(f"  '{test_word}' not found in Model 1's vocabulary.")
        print(f"\nTop 10 words similar to '{test_word}' in Model 2:")
        if sim2:
            for word, score in sim2:
                print(f"  {word}: {score:.4f}")
        else:
            print(f"  '{test_word}' not found in Model 2's vocabulary.")

    # OPTIONAL: Run new analogy tests.
    new_tests_set1, new_tests_set2 = run_optional_new_tests(model1)
    print("\nOptional New Analogy Tests (Set 1) Results for Model 1:")
    for a, b, c, expected, pred in new_tests_set1:
        print(f"  {a}:{b} :: {c} => Expected: {expected}, Predicted: {pred}")

    print("\nOptional New Analogy Tests (Set 2) Results for Model 1:")
    for a, b, c, expected, pred in new_tests_set2:
        print(f"  {a}:{b} :: {c} => Expected: {expected}, Predicted: {pred}")


if __name__ == '__main__':
    main()
