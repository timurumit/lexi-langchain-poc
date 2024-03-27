import os
import pinecone_datasets

from langchain.chat_models import ChatOpenAI
#from langchain.vectorstores import Pinecone
#from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings


def connect_to_pinecone(index_name):
    import pinecone

    #connect to pinecone using API key
    pinecone_index = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY") or 'b0cb3c41-9f37-4189-931d-d346c204fb59', 
                environment="us-west1-gcp")
    print("Pinecone connected")

    index = pinecone_index.Index(index_name)
    return index

#### initilises variables ####
#getting the api key from the environment variable
openAIkey = os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

index_name = 'lexi-dev-index-1'
text_field = "text"
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openAIkey)

index = connect_to_pinecone(index_name)

#initialise vectorstore object
vectorstore = PineconeVectorStore(index, embed_model,text_field)

query1 = "Apa UU yang mengatur perbankan?"

vectorstore.similarity_search(query1, k=3)
