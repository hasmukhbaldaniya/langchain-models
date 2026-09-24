from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)


class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")


parser = PydanticOutputParser(pydantic_object=Facts)


template = PromptTemplate(
    template="Give 3 fact about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


prompt = template.invoke({
    "topic": "Cricket"
})


result = model.invoke(prompt)


final_result = parser.parse(result.content)


print(final_result)

## chain

# chain = template | model | parser

# result = chain.invoke({
#     "topic": "black hole"
# })

# print(result)