from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()


# Define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai"
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):

    name: str = Field(
        description="Name of the person"
    )

    age: int = Field(
        gt=18,
        description="Age of the person"
    )

    city: str = Field(
        description="Name of the city the person belongs to"
    )


parser = PydanticOutputParser(
    pydantic_object=Person
)

template = PromptTemplate(
    template="Give 3 fact about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# prompt = template.invoke({
#     "topic": "Cricket"
# })

# print(prompt)


# result = model.invoke(prompt)


# final_result = parser.parse(result.content)


# print(final_result)

## chain

chain = template | model | parser

result = chain.invoke({
    "topic": "hockey"
})

print(result)