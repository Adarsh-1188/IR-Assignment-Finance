import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IRSystem:
    """Finance-domain Information Retrieval system using TF-IDF + cosine similarity."""

    def __init__(self, doc_dict):
        nltk.download("stopwords", quiet=True)
        self.raw_documents = doc_dict
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        self.processed_corpus = {
            doc_id: self._preprocess(text)
            for doc_id, text in self.raw_documents.items()
        }

        self.doc_ids = list(self.processed_corpus.keys())
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(
            list(self.processed_corpus.values())
        )

    def _preprocess(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        tokens = text.split()
        cleaned = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        return " ".join(cleaned)

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        processed_q = self._preprocess(query)
        query_vector = self.vectorizer.transform([processed_q])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            results.append({
                "Rank": rank,
                "Doc ID": self.doc_ids[idx],
                "Score": round(float(scores[idx]), 4),
                "Text": self.raw_documents[self.doc_ids[idx]]
            })

        return pd.DataFrame(results)
