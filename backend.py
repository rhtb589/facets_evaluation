from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
import sqlite3
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

app = FastAPI()

DB_PATH = "conversations.db"

# =========================
# LOAD FACETS
# =========================

facets = pd.read_csv("Facets.csv")
facets.columns = ["facet"]

facets["facet_clean"] = (
    facets["facet"]
    .astype(str)
    .str.lower()
    .str.strip()
    .apply(lambda x: re.sub(r"[^a-zA-Z ]", "", x))
)

# =========================
# EMBEDDINGS
# =========================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

facet_vectors = embedding_model.embed_documents(
    facets["facet_clean"].tolist()
)

facets['embeddings'] = facet_vectors

facet_vectors = np.array(
    facet_vectors,
    dtype="float32"
)

faiss.normalize_L2(facet_vectors)

index = faiss.IndexFlatIP(
    facet_vectors.shape[1]
)

index.add(facet_vectors)

# ====================== RUBRIC ======================
rubric = {
    "1": "Not present",
    "2": "Slightly present",
    "3": "Moderately present",
    "4": "Strongly present",
    "5": "Extremely present"
}
facets['score_rubric'] = json.dumps(rubric)

# ====================== CATEGORIZATION ======================
category_definitions = {
    "emotion": "emotional states, feelings, mood, affect, sentiment, expressiveness, empathy, emotional tone",
    "safety": "safety, harm prevention, risk, danger, toxicity, ethical concerns, responsible behavior",
    "pragmatics": "social language use, context awareness, tone appropriateness, politeness, sarcasm, implicature, common sense",
    "linguistic": "language quality, grammar, fluency, coherence, clarity, vocabulary, syntax, readability, conciseness"
}

category_names = list(category_definitions.keys())
category_texts = list(category_definitions.values())

# Embed categories once
category_embeddings_list = embedding_model.embed_documents(category_texts)
category_embeddings = np.array(category_embeddings_list, dtype="float32")

facet_embeddings = np.array(facets['embeddings'].tolist(), dtype="float32")

# L2 Normalization (vectorized)
facet_norms = np.linalg.norm(facet_embeddings, axis=1, keepdims=True)
category_norms = np.linalg.norm(category_embeddings, axis=1, keepdims=True)

facet_embeddings_norm = facet_embeddings / facet_norms
category_embeddings_norm = category_embeddings / category_norms

# Cosine similarity matrix (fully vectorized)
similarities = cosine_similarity(facet_embeddings_norm, category_embeddings_norm)

# Get best category index and confidence for each facet
best_category_idx = similarities.argmax(axis=1)
facets['category'] = [category_names[i] for i in best_category_idx]
facets['category_confidence'] = similarities.max(axis=1)

# =========================
# LLM
# =========================

llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

class FacetScore(BaseModel):
    selected_facets: list[str]
    categories: list[str]
    category_scores: dict
    score: int
    confidence: float
    reason: str

parser = JsonOutputParser(
    pydantic_object=FacetScore
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a conversation evaluation system. Available facets:{facets} Facet Categories: {categories}
        Scoring Rubric:{rubric}
        Instructions:
        1. Identify relevant facets.
        2. Group them by category.
        3. Score each category from 1-5.
        4. Produce an overall score.
        5. Return confidence.
        6. Explain reasoning.
        {format_instructions}
"""
    ),
    (
        "user",
        "{conversation}"
    )
])
structured_llm = llm.with_structured_output(
    FacetScore
)
chain = (
    prompt.partial(
        format_instructions=
        parser.get_format_instructions()
    )
    | structured_llm
    
)

# =========================
# DB INIT
# =========================

def init_db():

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation TEXT UNIQUE,
        score INTEGER,
        confidence REAL,
        reason TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# REQUEST MODEL
# =========================

class ConversationRequest(BaseModel):
    conversation: str

# =========================
# API
# =========================

@app.post("/evaluate")
def evaluate(req: ConversationRequest):

    query_embedding = embedding_model.embed_query(
        req.conversation
    )

    query_vector = np.array(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(query_vector)

    D, I = index.search(
        query_vector,
        k=20
    )

    retrieved_facets = (
        facets.iloc[I[0]]
        ["facet_clean"]
        .tolist()
    )
    retrieved_df = facets.iloc[
        I[0]
    ]

    retrieved_facets = (
        retrieved_df["facet_clean"]
        .tolist()
    )

    retrieved_categories = (
        retrieved_df["category"]
        .unique()
        .tolist()
)
    result = chain.invoke(
{
    "facets":
    "\n".join(
        retrieved_facets
    ),

    "categories":
    "\n".join(
        retrieved_categories
    ),

    "rubric":
    json.dumps(
        rubric,
        indent=2
    ),

    "conversation":
    req.conversation
}
)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
    """
    INSERT INTO conversations(
        conversation,
        score,
        confidence,
        reason
    )
    VALUES(
        ?, ?, ?, ?
    )
    """,
    (
        req.conversation,
        result.score,
        result.confidence,
        result.reason
    )
)
    conn.commit()
    conn.close()

    return result