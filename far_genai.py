from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import uvicorn

# Load the spacy model
nlp = spacy.load("en_core_web_md")

app = FastAPI()


class SimilarityRequest(BaseModel):
    baseline_response: str
    generated_response: str
    variance_limit: float


@app.post("/similarity/")
async def calculate_cosine_similarity(request: SimilarityRequest):
    try:
        # Create the doc objects
        doc1 = nlp(request.baseline_response)
        doc2 = nlp(request.generated_response)

        # Check if both texts have vectors
        if not doc1.has_vector or not doc2.has_vector:
            raise HTTPException(status_code=400, detail="One or both texts do not have a vector representation.")

        # Calculate cosine similarity
        similarity = doc1.similarity(doc2)

        # Determine pass or fail based on cutoff
        status = "pass" if similarity >= request.variance_limit else "fail"
        return {"cosine_similarity": similarity, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, port=8002)
