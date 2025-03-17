from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

#MONGO_URI = "mongodb://admin:adminpassword@localhost:27017/?replicaSet=rs0"
MONGO_URI = "mongodb://localhost:27017/?replicaSet=rs0"


try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5-second timeout
    client.admin.command("ping")  # Ping the MongoDB server
    print("✅ MongoDB connection successful!")
except ConnectionFailure as e:
    print(f"❌ MongoDB connection failed: {e}")
