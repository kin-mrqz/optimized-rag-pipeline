import os
import pandas as pd
import spacy
from rag_setup.get_chroma_db import load_vectorstore


def rag_system_setup():
    nlp_textcat = spacy.load("spacy_models/textcat_model")
    nlp_ner_wine = spacy.load("spacy_models/ner_wine_model")
    nlp_ner_food = spacy.load("spacy_models/ner_food_model")

    persist_dir = "chroma_db"
    if os.path.exists(persist_dir):
            vectorstore = load_vectorstore()
    else:
        print("No vectorstore found, create a new one.")
        vectorstore = None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    wine_data = "data/resources/wine_data.csv"
    food_data = "data/resources/food_data.csv"
    wine_df = pd.read_csv(wine_data)
    food_df = pd.read_csv(food_data)
    wine_json = wine_df.to_dict(orient="records")
    food_json = food_df.to_dict(orient="records")
    
    return retriever, nlp_textcat, nlp_ner_wine, nlp_ner_food, wine_json, food_json