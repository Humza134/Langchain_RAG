from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate environment variables
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found. Please set it in your environment variables.")

# Utility function to load and extract text from PDFs
def extract_text_from_pdfs(pdf_docs) -> str:
    """Extract and return text from a list of PDF files."""
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF files: {e}")

# Text splitting for chunking data
def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into smaller chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    
    if len(chunks) == 0:
        raise ValueError("No chunks were created. Check text splitting parameters.")
    
    return chunks

# Vector store creation and caching
def create_vector_store(text_chunks, api_key: str, save_path: str = "faiss_index"):
    """Create and save a vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(save_path)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {e}")

def load_vector_store(api_key: str, save_path: str = "faiss_index"):
    """Load a vector store from local storage."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

# Conversational chain creation
def create_conversational_chain() -> ChatGoogleGenerativeAI:
    """Create and return a conversational chain."""
    template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." 
    Don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=google_api_key, temperature=0.2)

    # Build the chain
    return {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt_template | llm | output_parser

def handle_user_query(user_question: str, vector_store, chain):
    """Process the user's question and return the context and response."""
    try:
        docs = vector_store.similarity_search(user_question)
        
        if not docs:
            raise ValueError("No relevant documents found in the vector store.")
        
        combined_docs = "\n\n".join([doc.page_content for doc in docs])  # Merge document content
        
        if not combined_docs.strip():
            raise ValueError("Combined context is empty. No relevant content to provide an answer.")
        
        response = chain.invoke({"context": combined_docs, "question": user_question})
        return docs, response
    except Exception as e:
        raise RuntimeError(f"Error handling user query: {e}")

# Streamlit application
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)
                    create_vector_store(text_chunks, api_key=google_api_key)
                    st.success("PDF processed and vector store created successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Input for user question
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        try:
            vector_store = load_vector_store(api_key=google_api_key)
            chain = create_conversational_chain()
            docs, response = handle_user_query(user_question, vector_store, chain)
            
            # Display the document similarity search results in an expandable section
            with st.expander("Document Similarity Search"):
                for doc in docs:
                    st.write(doc.page_content)
                    st.write("-----------------------")  # Separator for readability

            # Display the answer
            st.subheader("Answer:")
            st.write(response)

        except Exception as e:
            st.error(f"Error answering your question: {e}")

if __name__ == "__main__":
    main()


