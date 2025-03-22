from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import RAGPipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="RAG Business App API")
# Initialize the pipeline (assumes documents are in the "data" folder)
pipeline = RAGPipeline(data_path="data", rebuild_index=False)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        logging.info("Received query via API: %s", request.query)
        answer = pipeline.run(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error("Error processing query: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
