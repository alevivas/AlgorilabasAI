# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


load_dotenv()

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


def handle_pdf():
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = ChatOpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
        
def handle_website():
        
    st.title("Chat with websites")

    # sidebar
    with st.sidebar:
        website_url = st.text_input("Website URL")

    if website_url is None or website_url == "":
        st.info("Please enter a website URL")

    else:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)    

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)


# app config
st.set_page_config(page_title="Algorilabs Playground", page_icon="🤖")

# CSS for styling the sidebar and images
st.markdown(
    """
    <style>
    /* Change the background color of the sidebar */
    [data-testid="stSidebar"] {
        background-color: #010e17;
    }

    [data-testid="stHeader"] {
             background-color: #1a2a38;
    }
    
   [data-testid="stBottom"] > div {
    background-color: #1a2a38;
}
    
    .main {
        background-color: #1a2a38;
        color: white;
    }

      /* Change the color of all text elements to white */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: white;
    }

    /* Style the input boxes */
    input {
        background-color: #010e17;
        color: white;
        border: 1px solid #e9c46a;
        border-radius: 5px;
    }

    /* Style the text input box */
    input::placeholder {
        color: #e9c46a;
    }

    /* Style the "Settings" header */
    .stTextInput > div > label {
        color: white;
        font-weight: bold;
        font-size: 18px;
    }

    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 30px;
    }

    /* Style for the option buttons */
    .sidebar-option {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #46db9a;
        text-align: center;
    }
    .sidebar-option:hover {
        background-color: #f4a261;
        cursor: pointer;
    }
    .sidebar-option-selected {
        background-color: #f4a261;
        color: white;
    }
    .stButton>button {
        background-color: #46db9a;
        color: white;
        font-size: 22px !important;
        font-weight: bold;
        border-radius: 5px;
        margin-bottom: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #f4a261;
        color: white;
    }
      /* Estilo personalizado para el input de URL */
    .stTextInput label {
        font-size: 20px;
        color: #FFFFFF; /* Color blanco */
    }
    .stTextInput div {
        font-size: 18px;
        color: #FFFFFF; /* Color blanco */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to select an option and simulate radio button logic
def select_option(option):
    st.session_state.selected_option = option

# Sidebar with options
st.sidebar.image("assets/AlgoriLabs.jpeg", use_column_width=True)
st.sidebar.markdown('<div class="sidebar-title">Welcome to Algorilabs Playground</div>', unsafe_allow_html=True)

# Check the current selection
if "selected_option" not in st.session_state:
    st.session_state.selected_option = "Chat with a Website"

# Button 1: Chat with a Website
if st.sidebar.button("Chat with a Website", key="btn1"):
    select_option("Chat with a Website")

# Button 2: Chat with a PDF
if st.sidebar.button("Chat with a PDF", key="btn2"):
    select_option("Chat with a PDF")

# Displaying selected option in the main area
if st.session_state.selected_option == "Chat with a Website":
    st.image("assets/AIWebSite.png", width=300)
    handle_website()
elif st.session_state.selected_option == "Chat with a PDF":
    st.image("assets/AIPdf.png",  width=300)
    handle_pdf()