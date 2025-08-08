# ragapi/llm_chain.py
import os, glob, json, re
from dotenv import load_dotenv
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_perplexity import ChatPerplexity

load_dotenv()
os.environ['PPLX_API_KEY'] = os.getenv("ppxapi", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

@lru_cache
def get_retriever():
    """Load documents and build retriever (runs once per deployment)."""
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

    Steps:
    1. Parse the query for details (age, procedure, location, policy duration).
    2. Search the context for relevant clauses, rules, or exclusions.
    3. Decide claim status ('Approved' or 'Rejected').
    4. Provide justification referencing documents & clauses.
    5. Determine the 'Amount'. If rejected, use "Not Applicable".

    Return ONLY valid JSON with:
    {{
        "Decision": "...",
        "Amount": "...",
        "Justification": "...",
        "Overall": "..."
    }}

    <context>
    {context}
    </context>

    User's Query: {input}
    """,
    input_variables=["context", "input"],
)

def clean_json_output(text: str) -> dict:
    """Extract JSON from LLM output even if extra text is present."""
    try:
        # Try direct parse first
        return json.loads(text)
    except:
        # Fallback: extract JSON block with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from: {text}")

def analyze_query(user_query: str) -> dict:
    """Main RAG pipeline with lazy retriever."""
    retriever = get_retriever()
    context_docs = retriever.get_relevant_documents(user_query)
    context_text = "\n".join([d.page_content for d in context_docs])
    raw_output = llm.invoke(prompt.format(context=context_text, input=user_query))
    return clean_json_output(raw_output.content if hasattr(raw_output, 'content') else raw_output)
