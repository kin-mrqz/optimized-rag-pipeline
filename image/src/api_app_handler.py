import uvicorn
import main
from fastapi import FastAPI
from pydantic import BaseModel
from query_model import QueryModel
from mangum import Mangum
import os

app = FastAPI()
handler = Mangum(app)
main.rag_system_setup()

class SubmitQueryRequest(BaseModel):
    query: str

@app.get("/")
def index():
    return {"message": "Welcome to the API"}

@app.post("/submit-query")
def submit_query(request: SubmitQueryRequest) -> QueryModel:
    response = main.query_prompt(request.query)

    new_query = QueryModel(
        query_text = request.query,
        answer_text = response.response_text,
        variant_id = response.variant_id,
        is_descriptive = response.is_descriptive, # helps in formatting response
        sources = response.sources,  
        is_complete = True
    )
    try:
        new_query.put_item()
        print("✅ Query saved to DynamoDB")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save to DynamoDB: {str(e)}")
        print("   This is normal for local development without DynamoDB setup")
        # Continue without failing the request

    return new_query

if __name__ == "__main__":
    port = 8000
    print(f"Running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
