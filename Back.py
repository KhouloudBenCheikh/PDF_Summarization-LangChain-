from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import openAI 
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback  


def process_text(text):
   
    #process the given text by splitting it into chucks and converting them 
    #into embeddings to form a knowledge base
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    #load the Model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')