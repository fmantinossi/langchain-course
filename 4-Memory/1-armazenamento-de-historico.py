from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "demo-session"}}

response1 = conversation_chain.invoke({"input": "Olá, meu nome é Fernando... quem é você?"}, config=config)
print("Resposta 1:", response1.content)
print("-"*20)

response2 = conversation_chain.invoke({"input": "Você pode repetir o meu nome?"}, config=config)
print("Resposta 2:", response2.content)
print("-"*20)

response3 = conversation_chain.invoke({"input": "Com base no meu nome, você poderia me enviar uma frase motivacional?"}, config=config)
print("Resposta 3:", response3.content)
print("-"*20)