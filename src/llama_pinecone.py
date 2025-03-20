from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import os
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from src.__init__ import documents

from dotenv import load_dotenv
load_dotenv()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Create index if it doesn't exist
if pinecone_index_name not in pc.list_indexes().names():
    raise ValueError(f"Index {pinecone_index_name} does not exist")

# Get the index
index = pc.Index(pinecone_index_name)
vector_store = LangchainPinecone.from_texts(
    texts=[t.page_content for t in chunks],
    index_name=pinecone_index_name,
    embedding=embeddings,
)

# Attempted models that didn't work:
# 1. "daryl149/llama-2-7b-chat-hf" - Required special access/permissions
# 2. "meta-llama/Llama-2-7b-chat-hf" - Gated model, requires Meta approval
# 3. "TheBloke/Llama-2-7B-Chat-GGML" - GGML format not compatible with transformers
# 4. "NousResearch/Llama-2-7b-chat-hf" - Too large for most consumer GPUs

# Using smaller Llama-1 model that's open access and requires less memory
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                           device_map="auto",
                                           torch_dtype=torch.float16,
                                        #    trust_remote_code=True,
                                        #    load_in_8bit=True
                                           )

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

def generate_response(user_input):
    response = chain.invoke({"query": user_input, "return_only_outputs": True})
    return response["result"]
