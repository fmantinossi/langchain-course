from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system = ("system", "Você é um assistente que responde perguntas no estilo {style}.")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate(
    [system,user]
)

messages = chat_prompt.format_messages(style="formal",question="Me explique o que é langchain em no máximo 3 linhas.")

for msg in messages:
    print(f"{msg.type}: {msg.content}") 

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
result = model.invoke(messages)

print(result.content)