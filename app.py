from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma      
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import sys
import os
import numpy as np

COMPANY = "Enter the company name"
NAME = "Company name"
SYMPLE = ""
MODEL = "7B-chat" #the model
MODEL_PATH = "/Users/abdullahalharthi/Meta_llama/llama.cpp/models/{MODEL}/ggml-model-q4_0.gguf"
persist_directory = f"./VectorDatabase_{SYMPLE}_{MODEL}"


def creacting_vectordb(embedding_model):
    print("creating vector database ")
    loader = DirectoryLoader(f"./Data_{SYMPLE}")
    #Load document 
    documents = loader.load()
    #In order to split the document we need to import RecursiveCharacterTextSplitter from Langchain framework  
    #Init text splitter, define chunk size 1000 and overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #Split document into chunks
    texts = text_splitter.split_documents(documents)

    #persist_directory = f"./VectorDatabase_{SYMPLE}_13B"
    vectordb = Chroma.from_documents(documents =texts, embedding=embedding_model, persist_directory=persist_directory)
    #save document locally
    vectordb.persist()
        
def set_Chatbot():

    embedding = LlamaCppEmbeddings(model_path=MODEL_PATH)

    #Init loader
    if not os.path.exists(persist_directory):
        creacting_vectordb(embedding)

    print("Getting the model..")
    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llama_llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.2,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,

        )

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    chat_memory = ConversationBufferMemory(memory_key="history", input_key='query', output_key='result', return_messages=True)

    The_template = """
        <s>[INST] <<SYS>>

        You are a helpful, respectful and honest assistant name's AI_chatBot.
        AI_chatBot is helpful, kind, honest, friendly, good at writing and never fails to answer 
        {Customer_Name}'s requests immediately and with details and precision.
        AI_chatBot is a knowledgeable customer service representative at a {Company_Name}. 
        Answer only qyestion releated to {Company_Name}, If you are unsure about an answer, truthfully say "I don't know"
        Respond in short words.

        Respond to {Customer_Name}'s question, interacts as an AI assistant named AI_chatBot work for {Company_Name}.
        Respond in helpful, and friendly way. 

        {Customer_Name} just contacted you with a question or issue related to our products or services.
        Respond to the customer's inquiry in a helpful and professional manner.
        Use the context (delimited by CONTEXT ) 
        and the chat history (delimited by HISTORY) to answer the Coustmer questions.
                
        <CONTEXT>
        {context}
        
        <HISTORY>
        {history}

        <</SYS>>
        <USER QUESTION>
        {Customer_Name}: {question}
        [/INST]
        
    """

    

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=The_template,
    )

    print("Creating the chain")

    chain = RetrievalQA.from_chain_type(
        llm=llama_llm, 
        retriever=vectordb.as_retriever(), 
        memory=chat_memory, 
        verbose= False,  
        chain_type_kwargs={
             "verbose": True,
             "prompt": prompt.partial(Company_Name=COMPANY, Customer_Name=NAME),
             "memory": ConversationBufferMemory(
                 memory_key="history",
                 input_key="question"),
        }  
    )
    return chain

def app():

    chain = set_Chatbot()

    print("Start the app")
    while True:
        prompt = input('Enter new prompt: ')
        if 'exit' in prompt or 'quit' in prompt:
            break  
        
        getAnswer = chain.run({'query': f'{prompt}'})

#main process
if __name__ == "__main__":
    
    try:
         app()
    except Exception as e:
        print("Error while running the app!")
        print(e)

