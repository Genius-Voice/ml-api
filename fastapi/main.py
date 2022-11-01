from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI()

class User(BaseModel):
    text: str
    uuid: Optional[str]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def read_root():
    return {"Status": "Ok"}

@app.post("/predict/sentiment")
def predict_sentiment(user_input: User):
    # fill in your code here
    return {"user_input": user_input.text}