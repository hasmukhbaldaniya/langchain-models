from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = ["Kolkata is the capital of West Bengal", "Delhi is the capital of India", "Mumbai is the capital of Maharashtra", "Bangalore is the capital of Karnataka", "Chennai is the capital of Tamil Nadu"]
embedding_vector = embeddings.embed_documents(documents)
query = "What is the capital of India?"
query_vector = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_vector], embedding_vector)[0]
index, score =sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Top Document Score:", score)