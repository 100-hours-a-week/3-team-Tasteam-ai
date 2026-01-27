from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")

import json

data = json.load(open("data/test_data_sample.json", "r"))
print(data["restaurants"][0]["reviews"][0])





'''
client.create_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)
'''