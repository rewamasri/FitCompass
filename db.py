import os
import certifi

from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set. Add it to your .env file.")

mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
mongo_db = mongo_client["FitCompassDB"]

users_collection = mongo_db["users"]
comments_collection = mongo_db["comments"]
reviews_collection = mongo_db["workout_reviews"]
counters_collection = mongo_db["counters"]


def ensure_indexes():
    users_collection.create_index("id", unique=True)
    users_collection.create_index("username", unique=True)
    users_collection.create_index("email", unique=True)


def next_user_id():
    doc = counters_collection.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq"]