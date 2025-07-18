import uvicorn
import main
from fastapi import FastAPI
from pydantic import BaseModel
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
def submit_query(request: SubmitQueryRequest):
    response = main.query_prompt(request.query)
    return response

if __name__ == "__main__":
    port = 8000
    print(f"Running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
