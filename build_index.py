from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE = Path(__file__).parent
DATA = BASE / "data"
CHUNKS = DATA / "chunks.json"
INDEX = DATA / "faiss.index"
META = DATA / "faiss_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX))

    META.write_text(
        json.dumps(
            {"model": MODEL_NAME, "count": len(chunks), "chunks": chunks},
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Created {INDEX}")
    print(f"Created {META}")
    print("Knowledge base is ready.")

if __name__ == "__main__":
    main()
