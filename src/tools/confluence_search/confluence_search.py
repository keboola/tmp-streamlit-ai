from atlassian import Confluence
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os


def conflu_search():
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
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", max_tokens=512)
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
    
    
if __name__ == "__main__":
    conflu_search()