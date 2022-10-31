from fastapi import FastAPI
import time
import string
import re
import pickle
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# Load model and vectorizer
model = pickle.load(open("../models/svm", 'rb'))
vectorizer = pickle.load(open("../featurizers/sparse_features", 'rb'))

class UserInput(BaseModel):
    text: str
    uuid: Optional[str]
    
def text_preprocessor(sentence):
    """ 
    this function cleans a text
    param: string
    returns: string
    """
    # Lowercase
    sentence = sentence.lower()
    # Remove punctuations
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Remove white spaces
    sentence = re.sub(' +', ' ',sentence).strip()
    return sentence

def count_vectorizer(clean_text, vectorizer):
    """
    this function vectorizes a text
    params: string, vectorizer object
    returns: array
    """
    # Transform text to vector
    vectorized_text = vectorizer.transform([clean_text])
    return vectorized_text

def classification_model(text, clean_text, vectorized_text, model):
    """
    this function predicts the sentiment of a text
    params: string, string, array, model object
    returns: dict
    """
    # Predict sentiment
    prediction = model.predict(vectorized_text)[0]
    # Get class probabilities
    prediction_proba = model.predict_proba(vectorized_text)[0]
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
                  clean_text=clean_text,
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
def predict_sentiment(user_input: UserInput):
    start_time = time.time()
    # Text cleaner
    clean_text = text_preprocessor(user_input.text)
    # Feature_extraction
    vectorized_text = count_vectorizer(clean_text, vectorizer)
    # Classification model
    result = classification_model(user_input.text, clean_text, vectorized_text, model)
    # update execution time
    result.update(execution_time="%s seconds" % (time.time() - start_time))
    # Render result
    return result
