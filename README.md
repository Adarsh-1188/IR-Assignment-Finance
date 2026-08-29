# Finance Information Retrieval (IR) Project

## Overview
This project implements a finance/banking Information Retrieval system using the **Vector Space Model (VSM)** with **TF-IDF** weighting and **cosine similarity** ranking.

The project is based on a corpus of 10 finance/banking research abstracts.

## Pipeline
1. **Corpus setup** – stores 10 finance/banking documents.
2. **Preprocessing** – lowercasing, removal of non-alphabetic characters, English stop-word removal, and Porter stemming.
3. **TF-IDF** – converts the processed documents into a TF-IDF feature matrix.
4. **Query processing** – applies the same preprocessing to a user query.
5. **Ranking** – calculates cosine similarity and returns the most relevant documents.
6. **Evaluation** – reports vocabulary size and dimension reduction.
7. **OOP implementation** – `IRSystem` combines ingestion, preprocessing, vectorization, and search.

## Files
- `IR_Assignment_Finance.docx` – project documentation/report.
- `documents.py` – finance document corpus.
- `corpus.txt` – plain-text copy of the corpus.
- `ir_system.py` – reusable `IRSystem` class.
- `run_project.py` – runs preprocessing, TF-IDF, evaluation, and sample searches.
- `requirements.txt` – Python dependencies.
- `sample_output.txt` – representative output from the assignment documentation.
- `test_notes.txt` – basic verification notes.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the project
```bash
python run_project.py
```

The first run may download the NLTK English stop-word list.

## Sample Queries
The documentation demonstrates:
- `deep learning stock trading prediction`
- `neural network credit risk default`

## Reported Results
The assignment documentation reports:
- 10 documents in the corpus
- 69 unique TF-IDF vocabulary features
- TF-IDF matrix dimension: `(10, 69)`
- 11.54% vocabulary dimension reduction
- Top result for `deep learning stock trading prediction`: Doc 1 with a cosine similarity score of 0.8117
- Top result for `neural network credit risk default`: Doc 3 with a cosine similarity score of 0.7964
