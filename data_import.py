import os

from datasets import load_dataset

dataset = load_dataset('text',
    data_files='/Users/Administrator/Desktop/COMBINED ARTICLES.txt',
    split='train')

print(dataset[0])

#building the index and connecting to it
import pinecone

#connect to pinecone using API key
pinecone_index = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY") or 'b0cb3c41-9f37-4189-931d-d346c204fb59', 
              environment="us-west1-gcp")
print("Pinecone connected")

#to do finalise the index initialisation
import time

index_name = 'lexi-dev-index-1'

# if index_name not in pinecone_index.list_indexes():
#     index_spec = {
#         "dimension": 1536,
#         "deployment": "serverless",
#         "metric": "cosine"
#     }
#     pinecone_index.create_index(name=index_name, dimension=1536,spec=index_spec)

# while not pinecone_index.describe_index(index_name).status['ready']:
#     time.sleep(1)

index = pinecone_index.Index(index_name)

# index.describe_index_stats()
# {'dimension': 1536,
#  'index_fullness': 0.0,
#  'namespaces':{},
#  'total_vector_count':0}

#embedding the dataset
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
embed_model=OpenAIEmbeddings(model="text-embedding-ada-002")

#testing to embed the article text file
# Specify the path to your text file
file_path = '/Users/Administrator/Desktop/COMBINED ARTICLES.txt'

# Read the content of the file and store it in a variable
# with open(file_path, 'r') as file:
#     test_file_content = file.read()

# data = embed_model.embed_documents([test_file_content])

# print("Embedding done")
# len(data),len(data[0])
# print("here")

#adding vector to the database
# index.upsert(items=[('pdf_vector', dataset)])

# retrieved_vector = index.query(ids=['pdf_vector'])
# print(retrieved_vector)

from tqdm.auto import tqdm
#data = dataset.to_pandas()

batch_size = 100

# for i in tqdm(range(0, len(data), batch_size)):
#     i_end = min(len(data), i+batch_size)
#     batch = data.iloc[i:i_end]
#     ids = [f"{x['doi']}-{x['chunk-id']}" for i,x in batch.iterrows()]
#     texts = [x['chunk'] for _, x in batch.iterrows()]   
#     embeds = embed_model.embed_documents(texts)
#     metadata = [
#         {'text' : x['chunk'],
#          'source' : x['source'],
#          'title' : x['title']
#         }
#         for i,x in batch.iterrows()
#     ]

#     index.upsert(vectors=zip(ids,embeds,metadata))

index.upsert(items=[('pdf_vector', dataset)])

