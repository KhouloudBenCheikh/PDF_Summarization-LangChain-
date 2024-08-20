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
    embeddings = HuggingFaceBgeEmbeddings (model_name='sentence-transformers/all-MiniLM-L6-v2')

    #Create a FAISS index from the text chunks using the embeddings
    knwoledgeBase = FAISS.from_texts(chunks, embeddings)

    return knwoledgeBase 

def summarizer (pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""

        #Extract text form each page of the text
        for page in pdf_reader.pages:
            text+= page.extract_text or ""

        knowledgeBase = process_text(text)
        
        #Define the query for Summarization
        query= "Summarize the content of the uploaded PdF file in 3 to 5 lines"

        if query:
            #searching similarity
            docs=knowledgeBase.similarity_search(query)

            #Specify the Model to use for generating the summary
            OpenAIModel = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(mode=OpenAIModel, temperature=0.1)

            #load a question answering chain with the specific Model
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                return response
            
    