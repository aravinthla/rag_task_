# Resume RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot built with LangChain and Streamlit that allows you to upload bulk resume PDFs and search for specific candidates using natural language queries.

## Features

- 📄 **Bulk PDF Upload**: Upload multiple PDF files containing resumes
- 🔍 **AI-Powered Search**: Query for candidates using natural language
- 🤖 **OpenAI Integration** (Optional): Use GPT-3.5 for intelligent answer generation
- 🆓 **Free Alternative**: Works with local embeddings (no API key required)
- 📊 **Source Tracking**: View which resumes contain the information

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up OpenAI API key** (for enhanced LLM responses):
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Or enter it directly in the Streamlit app sidebar

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Configure settings** (in the sidebar):
   - Choose whether to use OpenAI (requires API key) or local embeddings
   - If using OpenAI, enter your API key

3. **Upload PDF files**:
   - Click "Upload PDF files containing resumes"
   - Select one or more PDF files
   - Click "Process Resumes"
   - Wait for processing to complete

4. **Search for candidates**:
   - Enter a natural language query in the search box
   - Examples:
     - "Find candidates with Python experience"
     - "Who has experience in machine learning?"
     - "Show me candidates with 5+ years of experience"
     - "Find someone with React and Node.js skills"
   - Click "Search" or press Enter
   - View the results and source documents

## How It Works

1. **PDF Processing**: The system extracts text from uploaded PDF files
2. **Text Chunking**: Resumes are split into smaller chunks for better retrieval
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
5. **Query Processing**: User queries are converted to embeddings and matched against stored resumes
6. **Answer Generation** (if OpenAI is enabled): GPT-3.5 generates intelligent answers based on retrieved context

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── rag_system.py       # RAG system implementation
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── chroma_db/         # Vector database (created automatically)
```

## Dependencies

- **streamlit**: Web interface
- **langchain**: RAG framework
- **langchain-community**: Community integrations
- **langchain-openai**: OpenAI integration
- **pypdf**: PDF text extraction
- **chromadb**: Vector database
- **sentence-transformers**: Local embeddings (free alternative)
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management

## Notes

- The first time you run the app, it will download the embedding model (~80MB)
- Processed resumes are stored in `chroma_db/` directory
- You can clear all data using the "Clear All Data" button in the sidebar
- Without OpenAI, the system returns raw retrieved text chunks
- With OpenAI, the system generates intelligent, summarized answers

## Troubleshooting

- **PDF not processing**: Ensure the PDF contains extractable text (not just images)
- **No results found**: Try rephrasing your query or check if resumes contain the information
- **OpenAI errors**: Verify your API key is correct and you have sufficient credits

## License

This project is open source and available for use.
