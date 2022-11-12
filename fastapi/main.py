from fastapi import FastAPI
import time
import pickle
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# Load model and vectorizer
model = pickle.load(open("../models/tfm_svm", 'rb'))

class User(BaseModel):
    text: str
    uuid: Optional[str]
    
def classification_model(text):
    """
    this function predicts the sentiment of a text
    params: string, string, array, model object
    returns: dict
    """
    # Predict sentiment
    prediction = model.predict([text])[0]
    # Get class probabilities
    prediction_proba = model.predict_proba([text])[0]
    # Get all classes
    prediction_classes = model.classes_
    # Create class ranking
    class_ranking = {classes:conf for classes, conf in zip(prediction_classes, prediction_proba)}
    # Sort class ranking
    class_ranking = dict(sorted(class_ranking.items(), key=lambda x: x[1], reverse=True))
    # Dictionary with empy values
    result = {}
    # update dictionay
    result.update(user_input=text,
                  prediction=prediction,
                  confidence=class_ranking[prediction],
                  class_ranking=class_ranking)

    return result

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def read_root():
    return {"Status": "Ok"}

@app.post("/predict/sentiment")
def predict_sentiment(user_input: User):
    start_time = time.time()
    # classification model
    result = classification_model(user_input.text)
    # update execution time
    result.update(execution_time="%s seconds" % (time.time() - start_time))
    # Render result
    return result