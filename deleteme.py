from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Connection string
uri = "mongodb://172.23.0.5:27017/?replicaSet=rs0"

# Connect to the MongoDB replica set
try:
    client = MongoClient(uri)

    # Check if the connection was successful
    print("Successfully connected to MongoDB.")

    # Access a database (e.g., 'testDB')
    db = client.testDB

    # Access a collection (e.g., 'testCollection')
    collection = db.testCollection

    # Perform a simple query (e.g., find all documents)
    result = collection.find()

    # Print the results
    for document in result:
        print(document)

except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
