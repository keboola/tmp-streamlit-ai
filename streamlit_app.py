import streamlit as st
import os
from streamlit_chat import message
#from src.agent import agent
from src.tools.confluence_search.confluence_search import conflu_search

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if not os.path.exists("./data"):
    os.mkdir("./data")
if not os.path.exists("data"):  
    os.mkdir("data")

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
        with st.spinner("Executing Query & Generating response..."):
            output = conflu_search(user_input).as_query_engine().query(user_input)
        st.success("Done!")
        st.balloons()
        st.write(output.response)
if __name__ == "__main__":
    main()