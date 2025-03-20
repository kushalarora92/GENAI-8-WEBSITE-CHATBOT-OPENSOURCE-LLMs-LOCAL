from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import os

from src.__init__ import documents
from dotenv import load_dotenv
load_dotenv()

persist_directory = "db"

model_name = "intfloat/e5-large-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Only create and store vectordb if directory doesn't exist or is empty
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    vectordb.persist()

# Load the existing vectordb
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# model = 'tiiuae/falcon-7b-instruct'
model_id = "tiiuae/falcon-rw-1b"  # ✅ 1B model, much lighter, Smallest Falcon model

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",  # Force CPU usage
    trust_remote_code=True,
)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="cpu",  # Force CPU usage
)

llm = HuggingFacePipeline(pipeline=pipeline)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)

def generate_response(user_input):
    response = qa_chain.invoke({"query": user_input, "return_only_outputs": True})
    return response["result"]  # or response.get("result", "I couldn't find an answer to that question.")
