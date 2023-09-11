#pip install pinecone-client langchain pypdf replicate
import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

api_key='1a07e0a3-d59b-4b01-b643-556e5210907e'
env='gcp-starter'
replicate='r8_2tIpI34I4Yu3mFyVsZeWElGqiyYs26g0WKbMM'


# Replicate API token
#os.environ['Default'] = replicate
os.environ['REPLICATE_API_TOKEN'] = replicate

# Initialize Pinecone
pinecone.init(api_key=api_key, environment=env)
# Load and preprocess the PDF document
loader = PyPDFLoader('/dbfs/FileStore/tables/META_Q1_2023_Earnings_Call_Transcript.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
texts[1]
documents
#Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()
#Set up the Pinecone vector database
index_name = "llama2"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)
#Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    # model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    #model="meta-llama/Llama-2-70b-chat-hf",
    input={"temperature": 0.5, "max_length": 25000} #here temp refers to randomness of the generated text
)
#Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True
)
chat_history = []
questions=["How AI has contributed to the growth of Reels ?"]
for i in questions:
    result = qa_chain({'question': i, 'chat_history': chat_history})
    print(i+"\n"+"Answer: "+result['answer']+"\n\n\n")
    chat_history.append((i, result['answer']))
chat_history
# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))
