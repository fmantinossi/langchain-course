from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def input_text(input: str) -> str:
    return input

question_template = PromptTemplate(
    input_variables=["question"],
    template="Olá, me explique em poucas linhas o que é o {question}?"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
chain = input_text | question_template | model
result = chain.invoke({"chain no langchain"})
print(result.content)