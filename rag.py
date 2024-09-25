import os
import re
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_groq import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure compatibility with async functions
nest_asyncio.apply()

# Load environment variables
load_dotenv()
HUGGINGFACE_KEY = os.environ.get('hugging_face_api')
LLAMA_CLOUD_API_KEY = os.environ.get('llama_parse_api')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Initialize the parser
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
)

# Function to check if the Chroma collection exists
def chroma_collection_exists(directory):
    # Check if the directory exists and contains files
    return os.path.exists(directory) and any(os.scandir(directory))

# Function to initialize and load the collection
def initialize_collection():
    # Load documents
    documents = parser.load_data(r"D:\WHISPER AI\Beginners-Guide-to-Selling-on-Amazon.pdf")

    # Extract text from documents
    all_text = " ".join([doc.text for doc in documents])

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    docs = text_splitter.create_documents([all_text])

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Set the persist directory
    persist_directory = "D:\WHISPER AI\CHROMA_EMBEDDINGS"
    collection_name = 'chroma_collection_1'

    # Check if the collection already exists
    if chroma_collection_exists(persist_directory):
        # Load the existing collection
        db = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embeddings)
        print(f"Collection '{collection_name}' loaded successfully.")
    else:
        # Create a new collection and persist it
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        db.persist()
        print(f"New collection '{collection_name}' created and persisted.")
    
    return db

# Initialize collection
db = initialize_collection()

# Define the LLM function
def model_v2_hf():
    llm = ChatGroq(api_key=GROQ_API_KEY, model="mixtral-8x7b-32768", temperature=0, max_tokens=1024, timeout=120, max_retries=2)
    return llm

# Initialize the LLM
mistral_llm = model_v2_hf()

# Define the prompt template
prompt = """
You are a friendly and informative bot that interacts with users in a conversational and engaging manner. 
When the user greets you, respond warmly with a greeting and inquire about their wellbeing before smoothly bringing the conversation back to the relevant topic. 
You engage in a polite and interactive dialogue, ensuring that your tone remains friendly and professional.

If the user's question is related to the specific topic or context, provide a helpful, non-technical, and easy-to-understand response. 
If the user's query is not related to the topic or context, politely steer them back to asking relevant questions while maintaining a positive and supportive tone.

Ensure the conversation flows naturally and is interactive, making the user feel heard and understood.

Context: {context}
Question: {question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

# Set up the retriever
retriever = db.as_retriever(
    search_kwargs={"k": 2}
)

# Set up the conversational chain
chatbot = ConversationalRetrievalChain.from_llm(
    llm=mistral_llm,
    retriever=retriever,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT},
    verbose=True
)

# Function to perform RAG and return the answer for a given query
def get_rag_answer(query):
    # Use the chatbot to answer the query
    result = chatbot({"question": query, "chat_history": []})
    return result['answer']

# Example usage
# query = "hi how are you?"
# answer = get_rag_answer(query)
# print("Answer:", answer)
