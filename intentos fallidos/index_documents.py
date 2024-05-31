import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PineconeException
import warnings

warnings.filterwarnings("ignore")


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Path to input documents
DATA_PATH1 = "fallout_content/PDFs"
DATA_PATH2 = "fallout_content/Nukapedia"
DATA_PATH3 = "fallout_content/YouTube_oxhorn"
DATA_PATH4 = "fallout_content/YouTube_spanish"

# Initialize embeddings
embeddings = OpenAIEmbeddings()


# Load documents from directory
def load_documents(path, extension):
    loader = DirectoryLoader(path, glob=extension)
    return loader.load()


# Split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# Generate data chunks from documents
def generate_data():
    documents = load_documents(DATA_PATH2, "*.txt")
    documents += load_documents(DATA_PATH1, "*.pdf")
    documents += load_documents(DATA_PATH3, "*.txt")
    documents += load_documents(DATA_PATH4, "*.txt")
    chunks = split_text(documents)
    return chunks


chunks = generate_data()

# Initialize Pinecone vector store and index documents
try:
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY, timeout=60
    )  # Adjust timeout as needed
    index_name = "fallout2"  # Define your Pinecone index name
    index = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name
    )
    print(f"Successfully indexed documents in {index_name}")
except PineconeException as e:
    print(f"An error occurred with Pinecone: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
