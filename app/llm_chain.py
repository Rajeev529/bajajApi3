import os, glob
from dotenv import load_dotenv
import logging
import json
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_perplexity import ChatPerplexity
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
os.environ['PPLX_API_KEY'] = os.getenv("ppxapi")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CACHING THE RETRIEVER ONCE DURING APPLICATION STARTUP ---
logger.info("Starting document loading and processing...")
try:
    docs = []
    # Use glob to find all PDF and DOCX files
    for path in glob.glob('bpdfs/*.pdf'):
        logger.info(f"Loading PDF: {path}")
        for d in PyPDFLoader(path).load():
            d.metadata['source_pdf'] = os.path.basename(path)
            docs.append(d)

    for path in glob.glob('bpdfs/*.docx'):
        logger.info(f"Loading DOCX: {path}")
        for d in Docx2txtLoader(path).load():
            d.metadata['source_docx'] = os.path.basename(path)
            docs.append(d)

    # Split documents into chunks
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings and the FAISS vector store
    logger.info("Creating embeddings and vector store...")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embedding)
    
    # Store the retriever in a global variable for immediate use
    CACHED_RETRIEVER = db.as_retriever(search_kwargs={"k": 4})
    logger.info("Document processing complete. Retriever is ready.")

except Exception as e:
    logger.error(f"Failed to load documents or create retriever: {e}")
    CACHED_RETRIEVER = None

# --- RAG CHAIN DEFINITION ---
llm = ChatPerplexity(temperature=0, model="sonar")

# Corrected Prompt Template
prompt = PromptTemplate(
    template="""
    You are an AI assistant specialized in analyzing policy documents.
    Your task is to analyze a list of questions and the provided context from insurance documents.
    For each question, you must provide a concise, direct answer based ONLY on the context.
    If the context does not contain the answer, state that the information is not available.

    The user will provide a list of questions. You must generate an answer for each question.
    Do not add any extra information or conversational filler.
    
    <context>
    {context}
    </context>

    Questions: {input}

    Please return a JSON object with a single key 'answers' which is a list of strings,
    where each string is the answer to one of the questions.
    Example:
    {{
      "answers": [
        "Answer 1",
        "Answer 2",
        "Answer 3"
      ]
    }}
    """,
    input_variables=["context", "input"],
)

parser = JsonOutputParser()

rag_chain = (
    {"context": CACHED_RETRIEVER, "input": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

def analyze_questions(questions: list) -> dict:
    if CACHED_RETRIEVER is None:
        logger.error("Retriever is not available. Cannot process query.")
        return {"error": "Internal server error. Document processing failed on startup."}
    
    # Pass the list of questions as a string to the chain
    questions_str = json.dumps(questions)
    return rag_chain.invoke(questions_str)