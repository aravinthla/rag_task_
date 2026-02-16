import streamlit as st
import os
import tempfile
from rag_system import ResumeRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "resumes_processed" not in st.session_state:
    st.session_state.resumes_processed = False
if "use_openai" not in st.session_state:
    st.session_state.use_openai = False

def initialize_rag_system(use_openai: bool = False):
    """Initialize the RAG system."""
    openai_api_key = None
    if use_openai:
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
        if not openai_api_key:
            st.error("OpenAI API key is required. Please set it in the sidebar or .env file.")
            return None
    
    try:
        return ResumeRAGSystem(use_openai=use_openai, openai_api_key=openai_api_key)
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI option
    use_openai = st.checkbox("Use OpenAI (GPT-3.5)", value=False, help="Requires OpenAI API key")
    
    if use_openai:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            help="Enter your OpenAI API key"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key
    
    st.session_state.use_openai = use_openai
    
    st.divider()
    
    # Clear data button
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear_vectorstore()
        st.session_state.rag_system = None
        st.session_state.resumes_processed = False
        st.success("Data cleared!")
        st.rerun()

# Main content
st.title("📄 Resume RAG Chatbot")
st.markdown("Upload bulk resume PDFs and query for specific candidates using AI-powered search.")

# Initialize RAG system if not already done
if st.session_state.rag_system is None:
    st.session_state.rag_system = initialize_rag_system(use_openai=st.session_state.use_openai)

# File upload section
st.header("📤 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF files containing resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF files. Each PDF can contain one or more resumes."
)

if uploaded_files and st.button("Process Resumes", type="primary"):
    if st.session_state.rag_system is None:
        st.error("Please configure the system in the sidebar first.")
    else:
        with st.spinner("Processing resumes... This may take a few moments."):
            try:
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    temp_files.append(temp_file.name)
                
                # Process resumes
                st.session_state.rag_system.process_resumes(temp_files)
                st.session_state.resumes_processed = True
                
                # Clean up temporary files
                for temp_file in temp_files:
                    os.unlink(temp_file)
                
                st.success(f"✅ Successfully processed {len(uploaded_files)} PDF file(s)!")
                
                # Show processed candidates
                candidates = st.session_state.rag_system.get_all_candidates()
                if candidates:
                    st.info(f"📋 Found resumes from: {', '.join(candidates)}")
                
            except Exception as e:
                st.error(f"Error processing resumes: {str(e)}")
                st.session_state.resumes_processed = False

# Query section
st.header("🔍 Search for Candidates")

if not st.session_state.resumes_processed:
    st.warning("⚠️ Please upload and process resumes first before searching.")
else:
    # Example queries
    st.markdown("**Example queries:**")
    st.markdown("- 'Find candidates with Python experience'")
    st.markdown("- 'Who has experience in machine learning?'")
    st.markdown("- 'Show me candidates with 5+ years of experience'")
    st.markdown("- 'Find someone with React and Node.js skills'")
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., Find candidates with Python and machine learning experience",
        key="query_input"
    )
    
    if st.button("Search", type="primary") or query:
        if query:
            with st.spinner("Searching through resumes..."):
                try:
                    # Search for candidates
                    use_llm = st.session_state.use_openai and st.session_state.rag_system.llm is not None
                    result = st.session_state.rag_system.search_candidates(query, use_llm=use_llm)
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result["answer"])
                    
                    # Display source documents
                    if result.get("source_documents"):
                        with st.expander("📚 View Source Documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f"*From: {doc.metadata.get('source', 'Unknown')}*")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.divider()
                    
                except Exception as e:
                    st.error(f"Error searching: {str(e)}")

# Footer
st.divider()
st.markdown("---")
st.markdown("**Resume RAG Chatbot** - Powered by LangChain and Streamlit")
