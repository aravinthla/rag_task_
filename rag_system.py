import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

class ResumeRAGSystem:
    def __init__(self, use_openai: bool = False, openai_api_key: Optional[str] = None):
        """
        Initialize the RAG system for resume processing.
        
        Args:
            use_openai: If True, use OpenAI embeddings and LLM. If False, use local models.
            openai_api_key: OpenAI API key (required if use_openai=True)
        """
        self.use_openai = use_openai
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize embeddings
        if use_openai and openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=openai_api_key
                )
            except TypeError:
                # Fallback for older versions
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=openai_api_key
                )
        else:
            # Use local embeddings (no API key needed)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.llm = None  # Will use retrieval only, no LLM generation
        
        self.vectorstore = None
        self.qa_chain = None
        self.persist_directory = "./chroma_db"
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def process_resumes(self, pdf_paths: List[str]) -> None:
        """
        Process multiple PDF resumes and create vector store.
        
        Args:
            pdf_paths: List of paths to PDF files
        """
        all_texts = []
        all_metadatas = []
        
        for pdf_path in pdf_paths:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create metadata for each chunk (include filename)
            filename = os.path.basename(pdf_path)
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({"source": filename, "pdf_path": pdf_path})
        
        # Create or update vector store
        if self.vectorstore is None:
            # Create new vector store
            self.vectorstore = Chroma.from_texts(
                texts=all_texts,
                metadatas=all_metadatas,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Add to existing vector store
            self.vectorstore.add_texts(
                texts=all_texts,
                metadatas=all_metadatas
            )
        
        # Create QA chain if using OpenAI
        if self.use_openai and self.llm:
            self._create_qa_chain()
    
    def _create_qa_chain(self):
        """Create the QA chain for question answering."""
        prompt_template = """Use the following pieces of context from resumes to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Provide a detailed answer about the candidate based on the resume information."""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        except Exception as e:
            # Fallback for different LangChain versions
            from langchain.chains import RetrievalQAWithSourcesChain
            try:
                self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": PROMPT}
                )
            except:
                # If all else fails, we'll use raw retrieval
                self.qa_chain = None
    
    def search_candidates(self, query: str, use_llm: bool = True) -> dict:
        """
        Search for candidates based on a query.
        
        Args:
            query: The search query (e.g., "candidate with Python experience")
            use_llm: If True and OpenAI is available, use LLM for answer generation.
                    If False, return raw retrieved documents.
        
        Returns:
            Dictionary with answer and source documents
        """
        if self.vectorstore is None:
            return {
                "answer": "No resumes have been processed yet. Please upload PDF files first.",
                "source_documents": []
            }
        
        if use_llm and self.qa_chain:
            # Use LLM for answer generation
            result = self.qa_chain({"query": query})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", [])
            }
        else:
            # Return raw retrieved documents
            docs = self.vectorstore.similarity_search(query, k=5)
            answer = "\n\n".join([doc.page_content for doc in docs])
            return {
                "answer": answer,
                "source_documents": docs
            }
    
    def get_all_candidates(self) -> List[str]:
        """Get list of all candidate filenames from processed resumes."""
        if self.vectorstore is None:
            return []
        
        # Get unique sources from metadata
        try:
            collection = self.vectorstore._collection
            if collection is None:
                return []
            
            results = collection.get()
            sources = set()
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata and "source" in metadata:
                        sources.add(metadata["source"])
            return list(sources)
        except AttributeError:
            # Alternative method: search with empty query to get all documents
            try:
                docs = self.vectorstore.similarity_search("", k=1000)
                sources = set()
                for doc in docs:
                    if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                        sources.add(doc.metadata['source'])
                return list(sources)
            except:
                return []
        except:
            return []
    
    def clear_vectorstore(self):
        """Clear the vector store and reset the system."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None
        self.qa_chain = None
