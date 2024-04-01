# lexi-langchain-poc
This repo is created for Lexi RAG proof of concept.

Upon initial research, few things are discovered.
- The need to store documents in a vector DB. This POC uses Pinecone.
- Utilising Langchain to connect to OpenAI (for both LLM and embeddings) and Pinecone

## Files
### main.py
Basic OpenAI Chat API utilisation 
### data_import_pdf.py
Using OpenAI and Pinecone APIs via Langchain to perform embeddings on a pdf file (single file only) and storing it in Pinecone DB
Pinecone API key is using Timur's private key, hardcoded in the python file
### RAG_to_chat.py
Again, using Langchain to take user query, perform similarity matching, and send the output into OpenAI LLM to generate user answer

## Additional notes
There are many Youtube and articles online on creating the above POC. However a lot of them are using deprecated libraries, and/or using pre-existing dataset as examples. It was hard to find an example that uses PDF files as the source knowledge, with a set of libraries that are still maintained. 
