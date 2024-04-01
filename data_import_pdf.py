from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# create a loader
loader = PyPDFLoader("/Users/Administrator/Desktop/Combined Articles.pdf")

# load your data
data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

# import libraries
from langchain.vectorstores import  Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or 'b0cb3c41-9f37-4189-931d-d346c204fb59'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

index_name = "lexi-dev-index-1"
 # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

query = "What are taxes payable in upstream oil and gas indistry"
query = "What is the most common form of security over tangible asset"
docs = docsearch.similarity_search(query)

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docs.as_retriever()
)

docs_string = [str(doc) for doc in docs]


docs
# Join the documents with "\n\n---\n\n" and append the query
augmented_query = "\n\n---\n\n".join(docs_string)+"\n\n-----\n\n"+query

##### OPENAI RETURN
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)

#creating a list of messages
messages = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content=augmented_query),
    #AIMessage(content="Hello!"),
]

responses = llm(messages)

print(responses.content)