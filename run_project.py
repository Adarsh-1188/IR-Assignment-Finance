import re
from documents import documents
from ir_system import IRSystem


def evaluate_pipeline(engine):
    raw_words = re.sub(
        r"[^a-zA-Z\s]", "", " ".join(engine.raw_documents.values()).lower()
    ).split()
    unique_raw = set(raw_words)
    processed_vocab = set(engine.vectorizer.get_feature_names_out())
    reduction = ((len(unique_raw) - len(processed_vocab)) / len(unique_raw)) * 100

    print("\n=== PIPELINE EVALUATION METRICS ===")
    print(f"Total Raw Corpus Word Count : {len(raw_words)}")
    print(f"Unique Raw Vocabulary Terms : {len(unique_raw)}")
    print(f"Processed Features (TF-IDF) : {len(processed_vocab)}")
    print(f"Vocabulary Dimension Reduction: {reduction:.2f}%")


engine = IRSystem(documents)

print("=== PREPROCESSING RESULTS ===")
for doc_id in list(documents)[:3]:
    print(f"[{doc_id}] Raw : {documents[doc_id]}")
    print(f"[{doc_id}] Preprocessed : {engine.processed_corpus[doc_id]}")
    print("-" * 65)

features = engine.vectorizer.get_feature_names_out()
print("\n=== TF-IDF MATRIX SUMMARY ===")
print(f"Total Unique Vocabulary Features : {len(features)}")
print(f"Matrix Dimension (Docs x Terms) : {engine.tfidf_matrix.shape}")

sample_terms = [
    term for term in ["bank", "financ", "credit", "custom", "fraud", "transact"]
    if term in features
]
tfidf_df = __import__("pandas").DataFrame(
    engine.tfidf_matrix.toarray(),
    index=engine.doc_ids,
    columns=features
)
print("\nSample TF-IDF Matrix Weights:")
print(tfidf_df[sample_terms].head(5).round(4))

print("\n=== QUERY SEARCH RESULTS ===")
print("Query: deep learning stock trading prediction")
print(engine.search("deep learning stock trading prediction", top_k=3).to_string(index=False))

evaluate_pipeline(engine)

print("\n=== CLASS SEARCH RESULTS ===")
print(engine.search("neural network credit risk default", top_k=2).to_string(index=False))
