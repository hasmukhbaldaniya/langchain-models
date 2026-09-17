from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

embedding_vector = embeddings.embed_documents(["Kolkata is the capital of West Bengal", "Delhi is the capital of India", "Mumbai is the capital of Maharashtra"])

print(str(embedding_vector))