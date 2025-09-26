from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

def prepare_input(payload: dict) -> dict:
    history = payload.get("history", [])
    trimmed_history = trim_messages(
        history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="user",
        include_system=True,
        allow_partial=False,

    )
    return {"input": payload.get("input", ""), "history": trimmed_history}

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

prepare = RunnableLambda(prepare_input)
chain = prepare | (prompt | llm)

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "demo-session"}}

response1 = conversation_chain.invoke({"input": "Olá, meu nome é Fernando... quem é você? Mas não mencione meu nome."}, config=config)
print("Resposta 1:", response1.content)
print("-"*20)

response3 = conversation_chain.invoke({"input": "Com base no meu nome, você poderia me enviar uma frase motivacional? mas não mencione meu nome."}, config=config)
print("Resposta 3:", response3.content)
print("-"*20)

response2 = conversation_chain.invoke({"input": "Você pode repetir o meu nome?"}, config=config)
print("Resposta 2:", response2.content)
print("-"*20)