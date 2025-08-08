# ragapi/llm_chain.py
import os, glob
from dotenv import load_dotenv
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_perplexity import ChatPerplexity

load_dotenv()
os.environ['PPLX_API_KEY'] = os.getenv("ppxapi")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

@lru_cache
def get_retriever():
    docs = []
    for path in glob.glob('bpdfs/*.pdf'):
        for d in PyPDFLoader(path).load():
            d.metadata['source_pdf'] = os.path.basename(path)
            docs.append(d)

    for path in glob.glob('bpdfs/*.docx'):
        for d in Docx2txtLoader(path).load():
            d.metadata['source_docx'] = os.path.basename(path)
            docs.append(d)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embedding)
    return db.as_retriever(search_kwargs={"k": 4})

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
        "Overall": generate overall ans like "Yes, knee surgery is covered under the policy"
    }}

    <context>
    {context}
    </context>

    User's Query: {input}
    """,
    input_variables=["context", "input"],
)

parser = JsonOutputParser()

rag_chain = (
    {"context": get_retriever(), "input": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

def analyze_query(user_query: str) -> dict:
    return rag_chain.invoke(user_query)
