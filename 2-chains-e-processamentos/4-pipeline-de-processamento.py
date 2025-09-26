from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

text_template = PromptTemplate(
    input_variables=["text"],
    template="Please, translate the following text to Portuguese: {text}\n"
)

summary_template = PromptTemplate(
    input_variables=["text"],
    template="Please, summarize the following text in a concise way: {text}\n"
)

llm_en = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

translate = text_template | llm_en | StrOutputParser()
summarize = {"text": translate} | summary_template | llm_en | StrOutputParser()
result = summarize.invoke({"text": "LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."})
print(result)