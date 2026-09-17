from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 512},
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Hello, how are you?")

print(result.content)