import streamlit as st
import os
from streamlit_chat import message
from src.agent import agent

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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