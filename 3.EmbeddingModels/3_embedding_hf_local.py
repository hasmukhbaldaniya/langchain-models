from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_vector = embeddings.embed_query("Kolkata is the capital of West Bengal")
embedding_vector = embeddings.embed_documents(["Kolkata is the capital of West Bengal", "Delhi is the capital of India", "Mumbai is the capital of Maharashtra", "Bangalore is the capital of Karnataka", "Chennai is the capital of Tamil Nadu"])

print(str(embedding_vector))