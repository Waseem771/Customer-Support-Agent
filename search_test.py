"""Quick retrieval test for the dummy index (uses the same dummy embedder).
For the real model, use build_real_index.py and set USE_REAL=True below."""
import json, re, hashlib, numpy as np, faiss, sys

USE_REAL = False   # True -> uses sentence-transformers all-MiniLM-L6-v2 (needs real index)
DIM = 384
STOP=set("the a an and or of to in on for is are be by with at as it its this that you your we our can if from will not any per".split())

def dummy_embed(text):
    toks=[t for t in re.findall(r"[a-z0-9]+",text.lower()) if t not in STOP]
    feats=toks+[a+"_"+b for a,b in zip(toks,toks[1:])]
    v=np.zeros(DIM,dtype="float32")
    for f in feats:
        h=int(hashlib.md5(f.encode()).hexdigest(),16)
        v[h%DIM]+= 1.0 if (h>>64)&1 else -1.0
        v[(h>>16)%DIM]+= 0.5 if (h>>65)&1 else -0.5
    n=np.linalg.norm(v); return v/n if n else v

def load():
    index=faiss.read_index("faiss.index")
    meta=json.load(open("chunks.json",encoding="utf-8"))
    return index, meta["chunks"]

def search(query, k=3):
    index, chunks = load()
    if USE_REAL:
        from sentence_transformers import SentenceTransformer
        q=SentenceTransformer("all-MiniLM-L6-v2").encode([query],normalize_embeddings=True).astype("float32")
    else:
        q=dummy_embed(query).reshape(1,-1)
    scores, ids = index.search(q, k)
    return [(float(s), chunks[i]) for s,i in zip(scores[0],ids[0])]

if __name__=="__main__":
    q=" ".join(sys.argv[1:]) or "how many days do I have to return a product"
    for s,c in search(q):
        print(f"{s:.3f} | {c['source_filename']} p{c['page_number']} | {c['text'][:90]}...")
