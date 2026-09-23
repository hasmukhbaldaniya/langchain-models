from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# 1st prompt -> detailed report
template = PromptTemplate(
    template="Write a detailed report on {topic}. Respond as a JSON object with a single key 'text' whose value is the full report.\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.format(topic="black hole")

result = model.invoke(prompt)

result = parser.parse(result.content)

print(result["text"])

print(type(result))


