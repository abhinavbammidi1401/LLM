#from langchain_openai import ChatOpenAI  # Open AI API
from langchain_core.prompts import ChatPromptTemplate  # Prompt template
from langchain_core.output_parsers import StrOutputParser  # Default output parser whenever an LLM model gives any response
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import streamlit as st  # UI
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith tracking (Observable)
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Defining Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Story Generator"),  # If user asks about any other unrelated topics, it will not answer. It will respond accordingly.
        ("user", "Question:{question}")
    ]
)

# Setting up models
# Ollama enables us to run large language models locally, automatically does the compression
# llm = ChatOpenAI(model="llm")
# llm = Ollama(model="llama2")  # Using Ollama and llama2 model
# outputParser = StrOutputParser()
# chain = prompt | llm | outputParser  # Defining chain - Combining

# Using Groq inference engine
# groqllm = ChatGroq(model="llama3-70b-8192", temperature=0)
groqApi = ChatGroq(model="gemma-7b-It", temperature=0)
outputparser = StrOutputParser()
chainSec = prompt | groqApi | outputparser

# Streamlit UI
#st.title("Story Generator using LangChain and Ollama API")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Profile", "Story Generator", "Question Answering"])

if page == "Profile":
    st.header("Abhinav.B ")
    st.write("Your average run-of-the-mil guy that just goes with the flow.")
    st.write("M.Sc AI/ML - 2nd year - Christ (Deemed to be University)")

elif page == "Story Generator":
    st.header("Story Generator")
    inputText = st.text_input("Give me a bombastic idea to write a story.")
    if inputText:
        st.write(chainSec.invoke({'question': inputText}))

elif page == "Question Answering":
    st.header("Question Answering")
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),  # This will change the behavior for question-answering
            ("user", "Question:{question}")
        ]
    )
    qa_chain = qa_prompt | groqApi | outputparser
    inputQuestion = st.text_input("Ask me anything. Ideas for a date? certainly! or maybe you need some help with your assignment? hell yeah!")
    if inputQuestion:
        st.write(qa_chain.invoke({'question': inputQuestion}))

