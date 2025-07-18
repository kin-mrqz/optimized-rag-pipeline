import os
import re
import json
import time
import random
import numpy as np
from functools import wraps
from statistics import quantiles
from rag_setup.get_rag_setup import rag_system_setup
from rag_setup.get_embedding import get_embedding_model

# Global Variables
llm = None
embd = get_embedding_model()
retriever = None
nlp_textcat = None
nlp_ner_food = None
nlp_ner_wine = None
wine_json = None
food_json = None
wine_log_times = {}
food_log_times = {}


# Utility Functions
def normalize(text):
    return re.sub(r'[^a-z]', '', str(text).lower())


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def extract_json_from_llm_output(output: str) -> dict:
    """
    Extracts and parses a JSON object from LLM output that may include Markdown formatting.
    Handles triple backticks, optional language labels, and excessive whitespace.
    """
    output = output.strip()

    # Match content between ```json ... ``` or just ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        json_str = output

    return json.loads(json_str)


# Stopwatch for evaluation 
def timed(log_times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            log_times[func.__name__] = round(end - start, 4)
            return result
        return wrapper
    return decorator


# Identify if the intent is to recommend food or wine
def recommend_wine(query:str):
    category = nlp_textcat(query)
    recommend_wine = category.cats["recommend_wine"]
    recommend_food = category.cats["recommend_food"]
    if recommend_wine > recommend_food:
        return True
    else:
        return False

def get_price(wine, price_field="price"):
    price_str = wine.get(price_field, "0")
    if not isinstance(price_str, str):
        price_str = str(price_str)
    # Remove commas, spaces, and common currency symbols
    cleaned = price_str.replace(",", "").replace(" ", "").replace("$", "").replace("HKD", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0

# for listed prices such as 150 - 200
def parse_average_price(value):
    if isinstance(value, str):
        value = value.strip()
        # Match range: "150-200", "150 - 200"
        range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", value)
        if range_match:
            low, high = map(float, range_match.groups())
            return (low + high) / 2
        # Match single number: "150"
        elif re.match(r"^\d+(\.\d+)?$", value):
            return float(value)
    return None

def sample_quartiles(wines, price_field="price", k_per_quartile=10):
    prices = [get_price(w, price_field) for w in wines]
    if not prices:
        return []
    
    q1, q2, q3 = quantiles(prices, n=4)
    buckets = {
        "Q1": [w for w in wines if get_price(w, price_field) <= q1],
        "Q2": [w for w in wines if q1 < get_price(w, price_field) <= q2],
        "Q3": [w for w in wines if q2 < get_price(w, price_field) <= q3],
        "Q4": [w for w in wines if q3 < get_price(w, price_field)]
    }

    sampled = []
    for group in buckets.values():
        if group:
            sampled.extend(random.sample(group, min(k_per_quartile, len(group))))
    return sampled                         
        

# LLM and Embedding Model Set-up
import boto3
from langchain_aws.llms.bedrock import BedrockLLM
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Configuration# change to bedrock llm
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
MODEL_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9
}

def setup_llm():
    """Set up and return the AWS Bedrock LLM."""
    # Set up AWS Bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    # Initialize Llama-3-70B-instruct model
    llm = BedrockLLM(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        model_kwargs=MODEL_PARAMS
    )

    return llm


# Prepare Schemas and Instruction Prompts
from typing import Optional
from pydantic import BaseModel, Field

# define simple schema as template, map doc.ent if present else None
class WineMetadata(BaseModel):
    """Schema for metadata filters"""
    wine_name: Optional[str] = Field(
        default = None, description = "Specific wine name mentioned in query"
        )
    max_price: Optional[float] = Field(
        default = None, description = "Specific max price mentioned in the query"
        )
    min_price: Optional[float] = Field(
        default = None, description = "Specific min price mentioned in the query"
        )
    wine_type: Optional[str] = Field(
        default = None, description = "Specific wine type (e.g., white, red, sparkling, etc.)"
        )

# Base Models for Reverse Pairings
class FoodMetadata(BaseModel):
    """Structured metadata filters for querying a food product database."""
    dish_name: Optional[str] = Field(
        default=None, description="Name of the dish (e.g., 'roast duck', 'brie cheese')."
    )
    min_price: Optional[float] = Field(
        default=None, description="Minimum price filter (inclusive)."
    )
    max_price: Optional[float] = Field(
        default=None, description="Maximum price filter (inclusive)."
    )
    food_type: Optional[str] = Field(
        default=None, description="Food type (e.g., 'fruit', 'pastry', 'vegetarian')."
    )
    course: Optional[str] = Field(
        default=None, description="Course (e.g., 'starter', 'main', 'dessert')."
    )

# Instruction Prompts
wine_system = """You are a master-level sommelier and wine data expert.
Your task is to create a hypothetical, ideal wine profile in the form of structured metadata based on a user's preference, question, or context.
This wine does not have to exist — it should represent the best possible match for what the user is looking for.
You must output a plausible, detailed JSON object that aligns with the schema below.
ONLY return the JSON. Do not explain, comment, or refer to the query.

JSON Format:
{
  "wine_name": "str",              // Specific wine name mentioned
  "winemaker": "str",              // Name of the winemaker or producer
  "vintage": "int",                // Vintage year (e.g. 2015)
  "country": "str",                // Country of origin
  "region": "str",                 // Region or appellation
  "wine_type": "str",              // Type of wine (e.g. red, white, rosé, sparkling)
  "wine_grapes": "str",            // Grape variety or blend (e.g. Merlot, Syrah)
  "occasion": "str",               // Occasion suitability (e.g. "wedding", "gift", "everyday")
  "body": "str",                   // Body type (e.g. "light", "full-bodied")
  "acidity": "str",                // Acidity level (e.g. "crisp", "low")
  "alcohol": "float",              // Alcohol content (as percentage, e.g. 13.5)
  "fruitiness": "str",             // Level of fruitiness (e.g. "dry", "juicy")
  "minerality": "str",             // Presence of mineral notes (e.g. "chalky", "flinty")
  "sweetness": "str",              // Sweetness level (e.g. "dry", "semi-sweet", "sweet")
}
"""

food_system = """You are a master-level chef and food data expert.
Your task is to create a hypothetical, ideal food profile in the form of structured metadata based on a user's preference, question, or context.
This food does not have to exist — it should represent the best possible match for what the user is looking for.
You must output a plausible, detailed JSON object that aligns with the schema below.
ONLY return the JSON. Do not explain, comment, or refer to the query.

JSON Format:
{
    "dish_name": "str",
    "food_type": "str",                  // e.g. 'meat', 'vegetarian', 'pastry'
    "course": "str",                // e.g. 'starter', 'main', 'dessert'
    "regional_pairing": "str",     // e.g. 'Provence', 'Piedmont'
    "sweetness": "str",            // 1–10
    "salitiness": "str",           // 1–10
    "acidity": "str",              // 1–10
    "sourness": "str",             // 1–10
    "umami": "str"                 // 1–10
}
"""

# Field Maps (i.e., which fields are valid metadata for filtering)
WINE_FIELD_MAP = {
    "wine_name": "Product Name",
    "min_price": "WS Retail Price",
    "max_price": "WS Retail Price",
    "wine_type": "Wine Type",
}

FOOD_FIELD_MAP = {
    "dish_name": "Product Name",
    "min_price": "Price",
    "max_price": "Price",
    "food_type": "Food Type",
    "course": "Course"
}


# MAIN FUNCTIONS
@timed(wine_log_times)
def wine_query_analyzer(query: str):
    """ Extracts structured wine metadata from a natural language query
    using a spaCy NER model and returns it as a WineMetadata object."""
    
    doc = nlp_ner_wine(query)
    all_labels = nlp_ner_wine.get_pipe("ner").labels
    result = {label: None for label in all_labels}
    
    for ent in doc.ents:
        if (ent.label_ == "min_price") or (ent.label_ == "max_price"):
            result[ent.label_] = float(ent.text)
        elif (ent.label_ == "wine_type"):
            if ent.text in ["red", "white", "sparkling"]:
                result[ent.label_] = ent.text
            else:
                result[ent.label_] = None 
        else:
            result[ent.label_] = ent.text
        
    return WineMetadata(**result)


@timed(food_log_times)
def food_query_analyzer(query: str):
    """ Extracts structured food metadata from a natural language query
    using a spaCy NER model and returns it as a FoodMetadata object."""

    doc = nlp_ner_food(query)
    all_labels = nlp_ner_food.get_pipe("ner").labels
    result = {label: None for label in all_labels}
    
    for ent in doc.ents:
        if (ent.label_ == "min_price") or (ent.label_ == "max_price"):
            result[ent.label_] = float(ent.text)
        else:
            result[ent.label_] = ent.text
        
    return FoodMetadata(**result)


@timed(wine_log_times)
def filter_wines(data, model_instance, field_map, max_results=20):
    """
    Given metadata filters constructed from user query using LLM,
    returns first n wine profiles that match. Also includes fallback logic
    """
    filters = model_instance.model_dump(exclude_none=True)
    results = []
    price_field = "WS Retail Price"

    # identify filter state
    has_filters = bool(filters)
    has_price = "min_price" in filters or "max_price" in filters
    has_type = "wine_type" in filters
    
    # case 1: have both filters
    if has_filters and ( has_price ):
        for wine in data:
            match = True
            for key, value in filters.items():
    
                if key not in field_map:
                    continue
                    
                if key in ("min_price", "max_price"):
                    wine_price = get_price(wine, price_field)
                    if key == "min_price" and wine_price < value:
                        match = False
                        break
                    if key == "max_price" and wine_price > value:
                        match = False
                        break
    
                else:
                    field = field_map.get(key)
                    if field not in wine:
                        match = False
                        break
                    wine_val = normalize(wine[field])
                    query_val = normalize(value)
                    if query_val not in wine_val:
                        match = False
                        break
    
            if match:
                results.append(wine)
                if len(results)== max_results:
                    break
        return results

    # case 2: filter only has wine_type (no price)
    elif has_type and not has_price:
        wine_type = normalize(filters["wine_type"])
        type_filtered = [w for w in data if normalize(w.get(field_map["wine_type"], "")) == wine_type]
        return sample_quartiles(type_filtered, price_field=price_field, k_per_quartile=4)

    else:
        type_buckets = {
            "red": [],
            "white": [],
            "sparkling": []
        }
        for w in data:
            wt = normalize(w.get(field_map.get("wine_type", "wine_type"), ""))
            if wt in type_buckets:
                type_buckets[wt].append(w)

        sampled = []
        sampled += sample_quartiles(type_buckets["red"], price_field, k_per_quartile=10)
        sampled += sample_quartiles(type_buckets["white"], price_field, k_per_quartile=10)
        sampled += sample_quartiles(type_buckets["sparkling"], price_field, k_per_quartile=10)
        return sampled[:max_results]


@timed(wine_log_times)
def create_wine_taste_profile(query: str):
    """
    Generates a hypothetical document embedding (HyDE) of 
    the ideal taste profile based on user query.
    """

    prompt = f"""{food_system}
User Query: {query.strip()}
"""
    
    response = llm.invoke(prompt)  
    
    return response


    
@timed(wine_log_times)
def generate_wine_recommendations(filtered, profile, top_k=5):
    """
    Performs similarity search between embeddings of (a) each of the filtered wines
    and (b) enriched taste profile from query, if present otherwise randomly 
    sample 3 wines (cheapest -> middle -> most expensive)
    """
    if len(filtered) == 0:
        return None
        
    if len(filtered) < top_k:
        return filtered
        
    if profile:
        embedded_wines = []
        for wine in filtered:
            content = "\n".join(f"{key}: {value}" for key, value in wine.items() if value)
            wine_embed = embd.embed_query(content)
            embedded_wines.append((wine, wine_embed))

        profile_embed = embd.embed_query(profile)
        scored_embed = [(wine, cosine_similarity(profile_embed, wine_embed)) for wine, wine_embed in embedded_wines]
        top_wines = sorted(scored_embed, key=lambda x: x[1], reverse=True)[:top_k]
        return [wine for wine, _ in top_wines]

    else:
        sorted_wines = sorted(filtered, key=lambda x: float(x.WS_Retail_Price))
        n = len(sorted_wines)
        if n == 0:
            return []
        step = max(1, n // top_k)
        sampled = [random.choice(sorted_wines[i:i+step]) for i in range(0, n, step)][:top_k]

        return sampled 


def ask_wine_ai(question, data, field_map):
    # start_main = time.time()
    parsed_query = wine_query_analyzer(question)
    filtered = filter_wines(data=data, model_instance=parsed_query, field_map=field_map)
    profile = create_wine_taste_profile(question)
    recommendations = generate_wine_recommendations(filtered, profile)

    # Print timing summary
    # total_time = round(time.time() - start_main, 4)
    # print("Timing summary:")

    # for name, duration in wine_log_times.items():
    #     print(f"  {name}: {duration} seconds")
        
    # print(f"  Total time: {total_time} seconds")
    
    return recommendations


@timed(food_log_times)
def filter_food(data, model_instance, field_map, max_results=20):
    """
    Given metadata filters constructed from user query using LLM,
    returns first n food taste profiles that match. Also includes fallback logic
    """
    filters = model_instance.model_dump(exclude_none=True)
    results = []
    price_field = "Price"

    # identify filter state
    has_filters = bool(filters)
    has_price = "min_price" in filters or "max_price" in filters
    has_type = "food_type" in filters
    has_course = "course" in filters  # implement course logic
    
    # case 1: have both price and type filters
    if has_filters and ( len(filters) > 1 or has_price ):
        for food in data:
            match = True
            for key, value in filters.items():
    
                if key not in field_map:
                    continue
                    
                if key in ("min_price", "max_price"):
                    food_price = get_price(food, price_field)
                    if key == "min_price" and food_price < value:
                        match = False
                        break
                    if key == "max_price" and food_price > value:
                        match = False
                        break
    
                else:
                    field = field_map.get(key)
                    if field not in food:
                        match = False
                        break
                    food_val = normalize(food[field])
                    query_val = normalize(value)
                    if query_val not in food_val:
                        match = False
                        break
    
            if match:
                results.append(food)
                if len(results)== max_results:
                    break
        return results

    # case 2: filter only has food_type (no price)
    elif has_type and not has_price:
        food_type = normalize(filters["food_type"])
        type_filtered = [w for w in data if normalize(w.get(field_map["food_type"], "")) == food_type]
        return sample_quartiles(type_filtered, price_field=price_field, k_per_quartile=4)

    # temporary fallback: choose randomly from beef, chicken, and pork dishes
    else:
        type_buckets = {
            "beef": [],
            "chicken": [],
            "pork": []
        }
        for w in data:
            wt = normalize(w.get(field_map.get("food_type", "food_type"), ""))
            if wt in type_buckets:
                type_buckets[wt].append(w)

        sampled = []
        sampled += sample_quartiles(type_buckets["beef"], price_field, k_per_quartile=10)
        sampled += sample_quartiles(type_buckets["chicken"], price_field, k_per_quartile=10)
        sampled += sample_quartiles(type_buckets["pork"], price_field, k_per_quartile=10)
        return sampled[:max_results]


@timed(food_log_times)
def create_food_taste_profile(query: str):
    """
    Generates a hypothetical document embedding (HyDE) of 
    the ideal taste profile based on user query.
    """
    
    prompt = f"""{food_system}
User Query: {query.strip()}
"""
    response = llm.invoke(prompt)
    return response
    

@timed(food_log_times)
def generate_food_recommendations(filtered, profile, top_k=5):
    """
    Performs similarity search between embeddings of (a) each of the filtered dishes
    and (b) enriched taste profile from query, if present otherwise randomly 
    sample 3 dishes (cheapest -> middle -> most expensive)
    """
    if len(filtered) == 0:
        return None
        
    if len(filtered) < top_k:
        return filtered
        
    if profile:
        embedded_food = []
        for food in filtered:
            content = "\n".join(f"{key}: {value}" for key, value in food.items() if value)
            food_embed = embd.embed_query(content)
            embedded_food.append((food, food_embed))

        profile_embed = embd.embed_query(profile)
        scored_embed = [(food, cosine_similarity(profile_embed, food_embed)) for food, food_embed in embedded_food]
        top_dishes = sorted(scored_embed, key=lambda x: x[1], reverse=True)[:top_k]
        return [dish for dish, _ in top_dishes]

    else:
        # Clean and sort
        filtered_clean = [
            item for item in filtered
            if parse_average_price(item["Price"]) is not None
        ]
        
        sorted_food = sorted(
            filtered_clean,
            key=lambda x: parse_average_price(x["Price"])
        )

        n = len(sorted_food)
        if n == 0:
            return []
        step = max(1, n // top_k)
        sampled = [random.choice(sorted_food[i:i+step]) for i in range(0, n, step)][:top_k]

        return sampled 


def ask_food_ai(question, data, field_map):
    # start_main = time.time()
    parsed_query = food_query_analyzer(question)
    filtered = filter_food(data=data, model_instance=parsed_query, field_map=field_map)
    profile = create_food_taste_profile(question)
    recommendations = generate_food_recommendations(filtered, profile)

    # Print timing summary
    # total_time = round(time.time() - start_main, 4)
    # print("Timing summary:")

    # for name, duration in food_log_times.items():
    #     print(f"  {name}: {duration} seconds")
        
    # print(f"  Total time: {total_time} seconds")
    
    return recommendations


def generate_recommendations(query: str):
    # Case 1: user looking for wine recommendations
    if recommend_wine(query):
        recommendations = ask_wine_ai(question=query, data=wine_json, field_map=WINE_FIELD_MAP)
        if recommendations == None:
            print("No recommendations found")
            return 
        
    # Case 2: user looking for food recommendations
    else: 
        recommendations = ask_food_ai(question=query, data = food_json, field_map = FOOD_FIELD_MAP)
        if recommendations == None:
            print("No recommendation found")
            return

    return recommendations


# RAG Instruction Prompt and Functions
reasoning_system = """
You are a world-class sommelier and culinary expert in wine and food pairings.
Given a user query and a list of dishes or wines, explain why each item pairs well.
Focus only on positive pairing notes. Keep each explanation to two sentences or less.
"""

def retrieve_context(query: str, retriever, k=5):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context


def generate_reasoning(query, item_list, context, llm):
    # start_time = time.time()
    results = []

    for item in item_list:
        prompt = f"""{reasoning_system}

User Query: {query}
Item: {item}
Context: {context}

Explain the pairing or recommendation:"""
        
        response = llm.invoke(prompt)
        results.append({"item": item, "explanation": response.strip()})

    # print(f"Time taken: {time.time() - start_time:.2f} seconds \n\n")
    return results


def ask_wine_secret(query, retriever, llm, top_k=5):
    # optional: can add functionality to access all database info about each recommended product
    recommendations_list = [product["Product Name"] for product in generate_recommendations(query)]
    context = retrieve_context(query, retriever, k=top_k)
    results = generate_reasoning(query=query, item_list=recommendations_list, context=context, llm=llm)
    return results
    

def query_prompt(user_query: Optional[str] = None):
    global llm, retriever, nlp_textcat, nlp_ner_wine, nlp_ner_food, wine_json, food_json
    
    llm = setup_llm()
    retriever, nlp_textcat, nlp_ner_wine, nlp_ner_food, wine_json, food_json = rag_system_setup()
    
    if not user_query:
        user_query = input("Ask a question about wine-food pairing: ")
    result = ask_wine_secret(user_query, retriever=retriever, llm=llm, top_k=5)

    return result


if __name__ == "__main__":
    rag_system_setup()
    result = query_prompt()
    print("\n\n".join([f"{product['item']}: {product['explanation']}" for product in result])) 

