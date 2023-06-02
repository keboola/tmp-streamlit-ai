import streamlit as st
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from atlassian import Confluence
import sys
import os
import langchain
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.langchain_helpers.agents import create_llama_chat_agent
from streamlit_chat import message
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI

from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["STORAGE_API_URL"] = st.secrets["STORAGE_API_URL"]
os.environ["STORAGE_API_TOKEN"] = st.secrets["STORAGE_API_TOKEN"]

llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

ROOT_DIR = "./data/"


@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

@tool
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"

async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))



from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain
    
    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError
      
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# !pip install duckduckgo_search
web_search = DuckDuckGoSearchRun()


confluence = Confluence(
 url=st.secrets["CONFLUENCE_URL"],
 username=st.secrets["CONFLUENCE_USERNAME"],
 password=st.secrets["CONFLUENCE_PASSWORD"])


# Function to translate the user's input into a CQL query
#def generate_cql_query_raw(input_text):
#    return f"text ~ '{input_text}'"


def generate_cql_query_keywords(input_text):
  # create a more advanced function to generate a CQL query. 
# It should utlize an LLM to extract keywords from the input 
# text and then use those to power the query
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512)
    keywords = llm.predict('Extract relevant keywords from the following:' + input_text)
    return f"text ~ '{keywords}'"

def construct_index(input_text):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

 
    st.write('Generating CQL query...')
    st.progress(0.1)
    
    #query_raw = generate_cql_query_raw(input_text)

    query_keywords = generate_cql_query_keywords(input_text)

    st.write('Query generated!')
    st.progress(0.2)
    st.write('Searching Confluence...')
    st.progress(0.3)
    
    pages_keywords = confluence.cql(query_keywords)
    
    pages = pages_keywords #pages_raw if len(pages_raw['results']) > len(pages_keywords['results']) else pages_keywords

    documents = []
   
    progress = 0.4
    for page in pages['results']:
        st.write(f"Found a potentially revevant page: {page['content']['title']}")
        progress += 0.01
        st.progress(progress)
        # Check the local directory to see if we already have the page's content
        if os.path.exists(f"data/{page['content']['id']}.txt"):
            st.write(f"Found the page's content locally: {page['content']['title']}")
            with open(f"data/{page['content']['id']}.txt", "r") as f:
                documents.append(f.read())
                f.close()
            continue
        
        # If we don't have the page's content, then get it from Confluence
        else:
            st.write(f"Getting the page's content from Confluence: {page['content']['title']}")
            content = confluence.get_page_by_id(page['content']['id'], expand='body.view')
            documents.append(content['body']['view']['value'])

             # add each page's content as a txt file in the data directory
            with open(f"data/{page['content']['id']}.txt", "w") as f:
                f.write(content['body']['view']['value'])
                f.close()

    # convert documents to a string
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    return index

def chatbot(input_text):


    tools = [
        web_search,
        WriteFileTool(root_dir="./data"),
        ReadFileTool(root_dir="./data"),
        process_csv,
        query_website_tool,
        Tool(
            name = "Conflu Index",
            func=lambda q: str(construct_index(input_text).as_query_engine().query(q)),
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
        output = chatbot(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

if __name__ == "__main__":
    main()
