from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    input_variables=["name"],
    template="Olá, meu nome é {name}, gostaria que você me dessa uma saudação apenas como um teste."
)

text = template.format(name="Fernando")
print(text)

