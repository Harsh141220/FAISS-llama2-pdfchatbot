import streamlit as st
from langchain.llms import Replicate
import os
import sys
from langchain.embeddings import HuggingFaceEmbeddings
import requests
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
import base64
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# App title
st.set_page_config(page_title="🦙💬 Eucloid data solutions Chatbot")
pdf = st.file_uploader("Upload your PDF", type='pdf')
chat_history=[]

api_key='1a07e0a3-d59b-4b01-b643-556e5210907e'
env='gcp-starter'
replicate_api="r8_BIgIL2qzL2O3F2dzTUkVt2AejeyYN6J1fomnq"
os.environ['REPLICATE_API_TOKEN'] = replicate_api


st.title('🦙💬 Eucloid data solutions Chatbot')

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    chat_history=[]
#st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    #string_dialogue = "You are an analyst. Your work is to refer the document/information provided to you and provide an answer. "
    '''for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"'''
    
    result = qa_chain({'question': prompt_input, 'chat_history': chat_history})
    return result['answer']
    

if pdf is not None:
    st.write(pdf.name)
    loader = PyPDFLoader(pdf.name)
    llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = 0.50
    max_length=25000

    pinecone.init(api_key=api_key, environment=env)
    embeddings = HuggingFaceEmbeddings()
    index_name = "llama2"
    index = pinecone.Index(index_name)
    documents = loader.load()   
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    llm2 = Replicate(
    model=llm,
    input={"temperature": temperature, "max_length": max_length } #here temp refers to randomness of the generated text
        )   
    qa_chain = ConversationalRetrievalChain.from_llm(llm2,vectordb.as_retriever(search_kwargs={'k': 2}),return_source_documents=True)
# Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
