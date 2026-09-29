"""Rebuild faiss.index with REAL all-MiniLM-L6-v2 embeddings from chunks.json.
    pip install sentence-transformers faiss-cpu
    python build_real_index.py
"""
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

meta = json.load(open("chunks.json", encoding="utf-8"))
model = SentenceTransformer(meta["embedding_model"])          # all-MiniLM-L6-v2
texts = [c["text"] for c in meta["chunks"]]
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
assert emb.shape[1] == meta["embedding_dimension"] == 384

index = faiss.IndexFlatIP(384)      # inner product on normalised vectors = cosine similarity
index.add(emb)
faiss.write_index(index, "faiss.index")

meta["note"] = "Real all-MiniLM-L6-v2 embeddings."
json.dump(meta, open("chunks.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Done:", index.ntotal, "vectors")
