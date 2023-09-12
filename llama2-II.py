import streamlit as st
from langchain.llms import Replicate
import os
import sys
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
#load api key lib
import base64

# App title
st.set_page_config(page_title="🦙💬 Eucloid data solutions Chatbot")
chat_history=[]
replicate_api="r8_SfExzEDw1tiyfpKl7ADFiAyaMu1rJfB1VE5m2"
os.environ['REPLICATE_API_TOKEN'] = replicate_api
pdf = st.file_uploader("Upload your PDF", type='pdf')
# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Eucloid data solutions Chatbot')
    
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.45, step=0.05)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=4096, value=512, step=8)

if pdf is not None:
    st.write(pdf.name)
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

    chunks = text_splitter.split_text(text=text)
    store_name = pdf.name[:-4]
    if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
    else:
            #embedding (Openai methods) 
            embeddings = HuggingFaceEmbeddings()
            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
                
            
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    chat_history=[]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "You are an analyst. Your work is to refer the document/information provided to you and provide an answer. "
    docs = vectorstore.similarity_search(query=query,k=3)
    llm2 = Replicate(
        model=llm,
    input={"temperature": temperature, "max_length": max_length, "top_p"=top_p } #here temp refers to randomness of the generated text
    )            
    chain = load_qa_chain(llm=llm2, chain_type= "stuff")
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    response = chain.run(input_documents = docs, question = f"{string_dialogue} {prompt_input} Assistant: ")
    return response
    


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

'''

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            
            #openai rank lnv process
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)'''
