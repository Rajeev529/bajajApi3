# ragapi/llm_chain.py
import os, glob
from dotenv import load_dotenv
from functools import lru_cache
import logging

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

# Set up logging to see progress during deployment
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CACHING THE RETRIEVER ONCE DURING APPLICATION STARTUP ---
# The logic is moved out of a function to ensure it's executed once
# when the module is loaded (i.e., when the server starts).
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
    # This will catch any errors during file loading or embedding creation
    # and prevent the server from crashing.
    logger.error(f"Failed to load documents or create retriever: {e}")
    CACHED_RETRIEVER = None

# --- RAG CHAIN DEFINITION ---
llm = ChatPerplexity(temperature=0, model="sonar")

prompt = PromptTemplate(
    template="""
    You are an AI assistant specialized in analyzing policy documents.
    Your task is to analyze a user's query and the provided context from a set of insurance documents.
    Based on this information, you must determine if an insurance claim would be 'Approved' or 'Rejected'.

    Follow these steps:
    1.  Parse the user's query to understand the core details (e.g., age, procedure, location, policy duration).
    2.  Search the provided context for relevant clauses, rules, or exclusions.
    3.  Based on the context, make a decision on the claim's status ('Approved' or 'Rejected').
    4.  Provide a detailed justification for your decision, referencing specific documents and clauses.
    5.  Determine the 'Amount'. If rejected, set this to 'Not Applicable'. If approved, provide a general status like 'Varies' or 'Enhanced Sum Insured' if applicable.

    Format your final response as a single JSON object with the following structure:
    {{
        "Decision": "Approved" or "Rejected",
        "Amount": "Not Applicable", "Varies", or a specific value if known,
        "Justification": "A detailed explanation referencing the documents and clauses.",
        "Overall": "Yes, knee surgery is covered under the policy"
    }}

    <context>
    {context}
    </context>

    User's Query: {input}
    """,
    input_variables=["context", "input"],
)

parser = JsonOutputParser()

# The rag_chain now uses the globally available CACHED_RETRIEVER directly.
rag_chain = (
    {"context": CACHED_RETRIEVER, "input": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

def analyze_query(user_query: str) -> dict:
    if CACHED_RETRIEVER is None:
        logger.error("Retriever is not available. Cannot process query.")
        return {"error": "Internal server error. Document processing failed on startup."}
    return rag_chain.invoke(user_query)

