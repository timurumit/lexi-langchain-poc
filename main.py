import os

from langchain.chat_models import ChatOpenAI

#getting the api key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#creating an instance of the chat model
chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

#creating a list of messages attributes default from langchain OpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)

#creating a list of messages
messages = [
    SystemMessage(content="You are a helpful assistant!"),
    HumanMessage(content="Tell me which planet are we on"),
    #AIMessage(content="Hello!"),
]

responses = chat(messages)

print(responses.content)