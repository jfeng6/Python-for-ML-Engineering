#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Score Headlines - Sentiment Analysis Script
Analyzes headlines and classifies them as Optimistic, Pessimistic, or Neutral.
"""

import sys
import os
import joblib
import datetime
from sentence_transformers import SentenceTransformer


# In[ ]:


def load_headlines(file_path):
    """Load headlines from a given file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            headlines = [line.strip() for line in file if line.strip()]
        return headlines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as error_message:  
        print(f"Error: {error_message}")
        sys.exit(1)


# In[ ]:


def vectorize_headlines(headlines, model):
    """Convert headlines to sentence embeddings."""
    return model.encode(headlines)


# In[ ]:


def classify_headlines(headlines, vectorizer, classifier):
    """Predict sentiment labels for headlines."""
    vectors = vectorizer.encode(headlines)
    predictions = classifier.predict(vectors)
    return list(zip(predictions, headlines))


# In[ ]:


def save_results(results, source):
    """Save the classified headlines to an output file."""
    today = datetime.date.today().strftime("%Y_%m_%d")
    output_filename = f"headline_scores_{source}_{today}.txt"

    with open(output_filename, "w", encoding="utf-8") as file:
        for label, headline in results:
            file.write(f"{label}, {headline}\n")

    print(f"Results saved to: {output_filename}")


# In[ ]:


def main():
    """Main function to execute the pipeline."""
    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <headlines_file> <source>")
        sys.exit(1)

    headlines_file = sys.argv[1]
    source = sys.argv[2]

    # Load the trained model and vectorizer
    classifier = joblib.load("svm.joblib")
    vectorizer = SentenceTransformer("all-MiniLM-L6-v2")

    # Load and process headlines
    headlines = load_headlines(headlines_file)
    results = classify_headlines(headlines, vectorizer, classifier)

    # Save the results
    save_results(results, source)


# In[ ]:


if __name__ == "__main__":
    main()

