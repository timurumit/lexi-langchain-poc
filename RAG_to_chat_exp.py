import openai
import os
from pinecone import Pinecone
#from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent

class LexiVector: 
    def __init__(self): 
        self.vectorstore = "geeksforgeeks"
        self.embeddings = 20

def setup_everything():
    # get api key from platform.openai.com
    openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
    api_key = os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY'
    environment = os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

    # configure client
    pc = Pinecone(api_key=api_key)
    embed_model = "text-embedding-ada-002"
    
    index = pc.Index(index_name)
    text_field = "text"

    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, model=embed_model)

    client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
    )

    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
        index, embeddings.embed_query, text_field
    )

    LexiVector.vectorstore =  vectorstore
    LexiVector.embeddings = embeddings

    return LexiVector

def ask(vectorstore,query,embeddings):
    #query = "UU perbankan itu nomor berapa?"

    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(
        query,  # our search query
        k=3  # return 3 most relevant docs
    )

    docs_string = [str(doc) for doc in docs]
    print("DEBUG message | Found in doc: ", docs_string)
    # Join the documents with "\n\n---\n\n" and append the query
    augmented_query = "\n\n---\n\n".join(docs_string)+"\n\n-----\n\n"+query
    print("DEBUG message | Augmented query: ", augmented_query)

    print("Your question: ", query)
    send_to_openai(augmented_query=augmented_query)

def send_to_openai(augmented_query):
    llm = ChatOpenAI(
        openai_api_key=openai.api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
    )
    
    #https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb#scrollTo=0qf5e3xf3ggq
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    qa.run(augmented_query)

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    # ##### OPENAI RETURN
    # from langchain.schema import(
    #     SystemMessage,
    #     HumanMessage,
    #     AIMessage
    # )

    # #creating a list of messages
    # messages = [
    #     SystemMessage(content="You are a helpful assistant!"),
    #     HumanMessage(content=augmented_query),
    #     #AIMessage(content="Hello!"),
    # ]

    # responses = llm(messages)
    # print("Answer:")
    # print(responses.content)

### Execute here ###
index_name = "lexi-dev-index-1"
vectorstore=setup_everything()

val = input("Enter your question: ")
agent(query)

# val = ""
# while val.lower() != "exit":
#     val = input("Enter your question: ")
#     if val.lower() != "exit":
#         ask(vectorstore.vectorstore, val, vectorstore.embeddings)
