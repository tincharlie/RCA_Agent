from pymongo import MongoClient
from config.settings import settings


client = MongoClient(
    settings.MONGO_URI
)

db =client[
    settings.MONGO_DB
]

def get_manual_collection():
    return db.manuals

def get_incident_collection():
    return db.incidents

