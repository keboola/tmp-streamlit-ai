import streamlit as st


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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

# Memory
embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# !pip install duckduckgo_search
web_search = DuckDuckGoSearchRun()


def agent(input_text):
    tools = [
        web_search,
        WriteFileTool(root_dir="./data"),
        ReadFileTool(root_dir="./data"),
        process_csv,
        query_website_tool,
        Tool(
            name = "Conflu Index",
            func=lambda q: str(conflu_search.construct_index(input_text).as_query_engine().query(q)),
            description="useful for when you want to answer questions about the internal docs from Confluence. The input to this tool should be a complete english sentence.",
            return_direct=True
    )
    # HumanInputRun(), # Activate if you want the permit asking for help from the human
]
    agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    #human_in_the_loop=True, # Set to True if you want to add feedback at each step.
    )
    #agent.chain.verbose = True
    
    
    response = agent.run([input_text])
    st.progress(1.0)
    st.balloons()
    return response