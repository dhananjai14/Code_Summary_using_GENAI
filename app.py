import os
import shutil
import streamlit as st
from langchain_chroma import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
from src.helper import repo_ingestion
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma


load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def create_vectors():
    documents = load_repo("repo/")
    text_chunks = text_splitter(documents)
    embeddings = load_embedding()
    #storing vector in choramdb
    vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
    return vectordb

def get_LLM(vectordb):
    llm = ChatOpenAI()
    memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)
    return qa



def main():
    st.set_page_config("CodeSummarizer")
    st.header("Python Code Summarizer Using GENAI")

    user_question = st.text_input("Ask a question from repository")
    with st.sidebar:
        st.title("Give the GitHub Repository address")
        user_repo_url = st.text_input("Enter Repo address")

        if st.button("Load Repository"):
            with st.spinner("Processing..."):
                repo_ingestion(user_repo_url)        
                vector_db = create_vectors()
                st.success("Done")

    if st.button("Send"):
        with st.spinner("Processing..."):
            vector_db = Chroma(persist_directory='./db', embedding_function=load_embedding())
            llm = get_LLM(vector_db)
            answer = llm(user_question)
            st.write(answer['answer'])
    if st.button('Clear repository database'):
        # shutil.rmtree(os.path.join(os.getcwd, 'db'))
        # shutil.rmtree(os.path.join(os.getcwd, 'repo'))
        st.write('Repo and DB cleared')

    

  




if __name__ == "__main__":
    main()