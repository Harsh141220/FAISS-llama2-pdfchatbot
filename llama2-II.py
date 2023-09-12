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

# App title
st.set_page_config(page_title="🦙💬 Eucloid data solutions Chatbot")
embeddings = HuggingFaceEmbeddings()
api_key='1a07e0a3-d59b-4b01-b643-556e5210907e'
env='gcp-starter'
pinecone.init(api_key=api_key, environment=env)
loader = PyPDFLoader("data/META-Q1-2023-Earnings-Call-Transcript.pdf")
print(loader)
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

#Set up the Pinecone vector database
index_name = "llama2"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
#Set up the Conversational Retrieval Chain

chat_history=[]
replicate_api="r8_SfExzEDw1tiyfpKl7ADFiAyaMu1rJfB1VE5m2"
os.environ['REPLICATE_API_TOKEN'] = replicate_api

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
    string_dialogue = "You are an analyst. Your work is to refer the document/information provided to you and provide an answer. If the information is not present in the pdf/text file, comment that you dont know. "
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    llm2 = Replicate(model=llm,input={"temperature": temperature, "max_length": max_length, "top_p":top_p,"repetition_penalty":1 }) #here temp refers to randomness of the generated text
    qa_chain = ConversationalRetrievalChain.from_llm(llm2,vectordb.as_retriever(search_kwargs={'k': 3}),return_source_documents=True)
    result = qa_chain({'question': f"{string_dialogue} {prompt_input}", 'chat_history': chat_history})
    chat_history.append((f"{prompt_input}", result['answer']))
    return result['answer']
    


# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
