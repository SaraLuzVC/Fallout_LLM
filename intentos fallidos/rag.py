from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
import os
import shutil
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import time
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeException

# Path to input documents
DATA_PATH1 = "fallout_content/PDFs"
DATA_PATH2 = "fallout_content/Nukapedia"
DATA_PATH3 = "fallout_content/YouTube_oxhorn"
DATA_PATH4 = "fallout_content/YouTube_spanish"

embeddings = OpenAIEmbeddings()
# prompt, model, parser


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Load documents from directory
def load_documents(path, extension):
    loader = DirectoryLoader(path, glob=extension)
    return loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def generate_data():
    documents = load_documents(DATA_PATH2, "*.txt")
    documents += load_documents(DATA_PATH1, "*.pdf")
    documents += load_documents(DATA_PATH3, "*.txt")
    documents += load_documents(DATA_PATH4, "*.txt")
    chunks = split_text(documents)
    return chunks

chunks = generate_data()


try:
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY, timeout=60
    )  # Adjust timeout as needed
    index_name = "fallout2"  # Define your Pinecone index name
    # Assuming PineconeVectorStore.from_documents is a valid method
    index = PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=index_name
    )

    print(f"Successfully indexed documents in {index_name}")
except PineconeException as e:
    print(f"An error occurred with Pinecone: {e}")
    index = None  # Ensure index is None if there was an error
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    index = None  # Ensure index is None if there was an error

chain = (
    {"context": index.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
chain.invoke("Who is Harold?")
chain.invoke("How to make Curie human?")


# Function to invoke the chain with a given question
def ask_question(question):
    return chain.invoke(question)


# Main loop
while True:
    # Get user input
    user_question = input("Please enter your question (or type 'exit' to quit): ")

    # Check if the user wants to exit the loop
    if user_question.lower() == "exit":
        print("Exiting the question loop. Goodbye!")
        break

    # Invoke the chain with the user's question and print the result
    response = ask_question(user_question)
    print(response)
