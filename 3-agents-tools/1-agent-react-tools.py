from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator_tool(expression: str) -> str:
    """Evaluate a simple math expression and returns the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mock web search tool that returns a canned response."""
    capitals = [
        {"country": "Brazil", "capital": "Brasília"},
        {"country": "Argentina", "capital": "Buenos Aires"},
        {"country": "United Kingdom", "capital": "London"},
        {"country": "France", "capital": "Paris"},
        {"country": "Germany", "capital": "Berlin"},
        {"country": "Japan", "capital": "Tokyo"},
        {"country": "China", "capital": "Beijing"},
        {"country": "Canada", "capital": "Ottawa"},
        {"country": "South Africa", "capital": "Pretoria"},
        {"country": "Australia", "capital": "Canberra"}
    ]
    for item in capitals:
        if item["country"].lower() in query.lower():
            return f"The capital of {item['country']} is {item['capital']}."

    return "I don't know the capital of that country."

llm = ChatOpenAI(model="gpt-5-mini", temperature=0.7, disable_streaming=True)
tools = [calculator_tool, web_search_mock]

prompt = PromptTemplate.from_template(
"""
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.
- Translate the final answer to Portuguese.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent_chain = create_react_agent(llm, tools, prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=False,
    handle_parsing_errors="Invalid format. Either provide and Action and Action Input, or a Final Answer.",
    max_iterations=5
)

print(agent_executor.invoke({"input": "What is the capital of Brazil?"}))
print(agent_executor.invoke({"input": "What is the capital of Argentina?"}))
print(agent_executor.invoke({"input": "What is the capital of the Iran?"}))
print(agent_executor.invoke({"input": "How much is 10+100?"}))









