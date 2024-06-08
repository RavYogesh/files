from fastapi import FastAPI, Body
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model (consider efficiency-optimized options)
model = SentenceTransformer("all-mpnet-base-v2")  # Example model, choose based on trade-off between accuracy and speed

app = FastAPI()

@app.post("/cosine_similarity")
async def calculate_similarity(sentence1: str = Body(...), sentence2: str = Body(...), threshold: float = Body(...)):
    # Preprocess sentences (optional, depending on model requirements)
    sentences = [sentence1, sentence2]
    # Consider tokenization, normalization, etc. if necessary

    # Encode sentences into vectors in a single batch for efficiency
    embeddings = model.encode(sentences, batch_size=len(sentences), convert_to_tensor=True)

    # Calculate cosine similarity efficiently using matrix multiplication
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    status = "pass" if similarity.item() >= threshold else "fail"

    return {"cosine_similarity": similarity.item(), "status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Adjust host and port as needed
