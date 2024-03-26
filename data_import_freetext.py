import os

from datasets import load_dataset
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#### loading the dataset ####
def process_pdf(file_path):
    # create a loader
    # load your data
    
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts

#### connecting to pinecone ####
# we are excluding the Pinecone index creation because we're on the free version
def connect_to_pinecone(index_name):
    import pinecone

    #connect to pinecone using API key
    pinecone_index = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY") or 'b0cb3c41-9f37-4189-931d-d346c204fb59', 
                environment="us-west1-gcp")
    print("Pinecone connected")

    index = pinecone_index.Index(index_name)

#### embedding the dataset ####
from langchain.embeddings.openai import OpenAIEmbeddings
embed_model=OpenAIEmbeddings(model="text-embedding-ada-002")

def create_embeddings(text):
    embeddings_list = []
    for text in texts:
        res = openai.Embedding.create(input=[text], engine=MODEL)
        embeddings_list.append(res['data'][0]['embedding'])
        return embeddings_list
data = embed_model.embed_documents(texts)

#### running the functions ####
file_path = '/Users/Administrator/Desktop/Combined Articles.pdf'
texts = process_pdf(file_path)
create_embeddings(texts)
index_name = 'lexi-dev-index-1'
connect_to_pinecone(index_name)

#### upserting the embeddings into the Pinecone index ####
index.upsert(vectors=[(id, embedding) for id, embedding in zip("/Users/Administrator/Desktop/Combined Articles.pdf", data)])
