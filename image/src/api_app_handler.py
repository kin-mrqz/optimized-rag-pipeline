import uvicorn
import main
from fastapi import FastAPI
from pydantic import BaseModel
from query_model import QueryModel
from mangum import Mangum

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
        # sources = response.sources,
        sources = response.sources,  # Placeholder for sources, not yet implemented
        is_complete = True
    )
    new_query.put_item()

    return new_query

if __name__ == "__main__":
    port = 8000
    print(f"Running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
