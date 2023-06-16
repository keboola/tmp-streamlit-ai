from typing import Callable
from langchain import LLMChain
import streamlit as st
#Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, AgentExecutor, initialize_agent, LLMSingleActionAgent, AgentOutputParser
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.prompts import StringPromptTemplate

#Memory Imports 
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun
from langchain.schema import Document

# Llama Index Imports
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
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
        #web_search,
        WriteFileTool(root_dir="./data"),
        ReadFileTool(root_dir="./data"),
        #process_csv,
        #query_website_tool,
        Tool(
            name = "Conflu Index",
            func=lambda q: str(conflu_search(input_text).as_query_engine().query(q)),
            description="useful for when you want to answer questions about the internal docs from Confluence. The input to this tool should be a complete english sentence. do not use this tool with the same input/query",
            return_direct=True
    )
    ]       
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vector_store.as_retriever()

    def get_tools(query):
        docs = retriever.get_relevant_documents(query)
        return [tools[d.metadata["index"]] for d in docs]
    
    # Set up the base template
    template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

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

    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

    Question: {input}
    {agent_scratchpad}"""
    from langchain.prompts import StringPromptTemplate

    from typing import Callable
    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        ############## NEW ######################
        # The list of tools available
        tools_getter: Callable
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            ############## NEW ######################
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            return self.template.format(**kwargs)
        
    prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
    )
    # HumanInputRun(), # Activate if you want the permit asking for help from the human
   # TODO: [AIS-74] Investigate other agent types
    auto_agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
    #human_in_the_loop=True, # Set to True if you want to add feedback at each step.
    )
    #agent.chain.verbose = True

    from langchain.prompts import StringPromptTemplate
    from langchain import OpenAI, SerpAPIWrapper, LLMChain
    from typing import List, Union
    from langchain.schema import AgentAction, AgentFinish
    import re
    class CustomOutputParser(AgentOutputParser):
    
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    single_action_agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tools
)

    super_simple_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    single_action_agent_executor = AgentExecutor.from_agent_and_tools(agent=single_action_agent, tools=tools, verbose=True)

    response = auto_agent.run(input_text)
    st.progress(1.0)
    st.balloons()
    return response