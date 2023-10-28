import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

# Sidebar contents
with st.sidebar:
    st.title('🗨️ LLM Chat App 🗨️')
    st.write("Welcome to the LLM Chat App powered by LangChain and OpenAI!")
    st.write("About this App:")
    st.markdown(
        "This app allows you to chat with a language model powered by LLM. "
        "Ask questions and get answers from a large corpus of text data. "
        "Try it out and explore the capabilities of modern language models."
    )
    st.markdown("[Streamlit](https://streamlit.io/)")
    st.markdown("[LangChain](https://python.langchain.com/)")
    st.markdown("[OpenAI LLM](https://platform.openai.com/docs/models)")
    st.write("Made by [Sampik Kumar Gupta](https://www.linkedin.com/in/sampik-gupta-41544bb7/)")

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def main():
    st.header("Ask any questions from your PDF by uploading it Here")
    
    # Initialize or retrieve conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Upload pdf
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Load PDF and extract text
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)

        # Embeddings and VectorStore
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions or queries
        query = st.text_input("Ask a question about your file:")

        if query:
            # Add user's query to the conversation with line breaks
            st.session_state.conversation.append(f"You: {query}\n")

            # Get previous conversation context
            context = "\n".join(st.session_state.conversation)
            st.write("Conversation:")
            st.write(context)

            # Search for answers
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Pass the entire conversation context to the chatbot
            full_context = "\n".join(st.session_state.conversation)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query, conversation=full_context)

            # Add chatbot's response to the conversation with line breaks
            st.session_state.conversation.append(f"Chatbot: {response}\n")
            
            st.write("Chatbot:")
            st.write(response)

if __name__ == '__main__':
    main()
