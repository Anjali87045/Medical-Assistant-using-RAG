from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import torch

from app.model import load_model_and_pipeline

# Configuration
DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

class RAGService:
    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.vectordb = None

    def initialize(self):
        """
        Initializes the Model and RAG chain. 
        This is heavy and should be done on startup.
        """
        # 1. Load LLM
        hf_pipeline = load_model_and_pipeline()
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # 2. Load Embeddings
        # Check device for embedding model
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs)

        # 3. Load VectorDB
        print(f"Loading ChromaDB from {DB_DIR}...")
        self.vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

        # 4. Create QA Chain
        retriever = self.vectordb.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        print("RAG Service initialized.")

    def answer_question(self, question: str):
        if not self.qa_chain:
            raise RuntimeError("RAG Service is not initialized.")
        
        return self.qa_chain(question)
