import streamlit as st
import os
from streamlit_chat import message
from pydantic import Field

#Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.docstore.document import Document
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain

#Memory Imports 
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun

# Llama Index Imports
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.langchain_helpers.agents import create_llama_chat_agent

# src Imports
from src.tools.confluence_search.confluence_search import conflu_search
from src.tools.process_csv import process_csv
from src.tools.query_website import WebpageQATool
from src.agent import agent

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["STORAGE_API_URL"] = st.secrets["STORAGE_API_URL"]
os.environ["STORAGE_API_TOKEN"] = st.secrets["STORAGE_API_TOKEN"]


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
ROOT_DIR = "./data/"

def main():
    st.title("Keboola Conflu AI Chatbot")
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    def get_text():
        input_text = st.text_input("You: ", "Answer the following question: What is the process for a customer to set up BYODB? Provide all relevant citations to confluence docs", key="input")
        return input_text


    user_input = get_text()

    if user_input:
        output = agent(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

if __name__ == "__main__":
    main()