from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    input_variables=["name"],
    template="Olá, meu nome é {name}, você poderia me dizer a origem do meu nome?"
)
model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
chain = LLMChain(llm=model, prompt=template)
result = chain.run("Fernando")
print(result)


question_template = PromptTemplate(
    input_variables=["question"],
    template="Olá, me explique em poucas linhas o que é o {question}?"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
chain = question_template | model
result = chain.invoke({"question": "langchain"})
print(result.content)