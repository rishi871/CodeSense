# app.py (or codesense_app.py)

import streamlit as st
import os
import shutil
import logging
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document # Used for type hinting
import google.generativeai as genai

# --- Configuration ---
DEFAULT_CODE_DIR = "./my-java-project" # CHANGE THIS if your code is elsewhere
VECTORSTORE_DIR = "./java_vectorstore_gemini" # Directory for ChromaDB files
# Google AI Model Names (Ensure these are available in your region/tier)
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_LLM_MODEL = "gemini-2.5-pro-exp-03-25" # Or "gemini-1.5-flash-latest", "gemini-pro"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LangChain & Indexing Functions ---

# Modified to accept api_key
@st.cache_resource(show_spinner="Loading and Indexing Java Code (Gemini)...")
def create_or_load_vectorstore(
    code_dir: str,
    vectorstore_path: str,
    embedding_model_name: str,
    api_key: str, # <<< Added API key argument
    force_reindex: bool = False
) -> Chroma | None:
    """
    Loads Java code, splits it, creates Gemini embeddings (using the provided API key),
    and stores/loads them in ChromaDB.
    Returns the Chroma vector store instance or None if an error occurs.
    """
    vector_store = None
    embeddings = None

    # Check if API key was passed
    if not api_key:
        st.error("API Key is missing. Cannot initialize embeddings.")
        logging.error("API Key is missing in create_or_load_vectorstore.")
        return None

    try:
        # Pass the API key directly to the constructor <<< FIX
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_key
        )
        logging.info(f"Initialized GoogleGenerativeAIEmbeddings with model: {embedding_model_name}")
    except Exception as e:
        # Catch errors related to API key/permissions during embedding initialization
        st.error(f"Failed to initialize Google Embeddings. Error: {e}. Make sure your API key is valid and has permissions.")
        logging.error(f"Failed to initialize Google Embeddings: {e}", exc_info=True)
        return None # Can't proceed without embeddings

    # --- Load Existing or Re-index ---
    if os.path.exists(vectorstore_path) and not force_reindex:
        try:
            st.info(f"Loading existing vector store from {vectorstore_path}")
            logging.info(f"Loading existing vector store from {vectorstore_path}")
            # Use the already initialized embeddings object
            vector_store = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            st.sidebar.success("Vector store loaded.")
            logging.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            st.warning(f"Failed to load existing vector store: {e}. Will attempt to re-index.")
            logging.warning(f"Failed to load existing vector store from {vectorstore_path}: {e}", exc_info=True)
            # Clean up potentially corrupted directory before re-indexing
            try:
                shutil.rmtree(vectorstore_path)
                logging.info(f"Removed potentially corrupted directory: {vectorstore_path}")
            except Exception as rm_err:
                st.error(f"Error removing corrupted vector store directory: {rm_err}")
                logging.error(f"Error removing corrupted directory {vectorstore_path}: {rm_err}", exc_info=True)
                return None # Stop if we can't clean up

    elif os.path.exists(vectorstore_path) and force_reindex:
         st.info(f"Re-indexing: Removing existing vector store at {vectorstore_path}")
         logging.info(f"Re-indexing: Removing existing vector store at {vectorstore_path}")
         try:
            shutil.rmtree(vectorstore_path)
         except Exception as e:
            st.error(f"Error removing vector store directory for re-indexing: {e}")
            logging.error(f"Error removing vector store directory {vectorstore_path} for re-indexing: {e}", exc_info=True)
            return None # Stop if we can't remove the old index

    # --- Create New Index ---
    if not os.path.exists(code_dir):
        st.error(f"Error: Code directory '{code_dir}' not found.")
        logging.error(f"Code directory '{code_dir}' not found.")
        return None

    st.info(f"Creating new vector store from code in {code_dir}")
    logging.info(f"Creating new vector store from code in {code_dir}")
    try:
        # 1. Load .java files
        loader = DirectoryLoader(
            code_dir,
            glob="**/*.java", # Recursive search for .java files
            loader_cls=TextLoader, # Use TextLoader for source code
            show_progress=True,
            use_multithreading=True, # Speed up loading
            loader_kwargs={'autodetect_encoding': True} # Handle potential encoding issues
        )
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} raw documents.")

        if not documents:
            st.warning("No .java files found in the specified directory. The index will be empty.")
            logging.warning(f"No .java files found in {code_dir}.")
            return None

        # 2. Split documents into chunks suitable for embedding
        java_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=1500, chunk_overlap=150
        )
        split_docs = java_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        # Optional: Add filename for easier source tracking
        for doc in split_docs:
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

        # 3. Create Chroma vector store and persist
        st.info(f"Creating embeddings using Google AI '{embedding_model_name}' and storing in Chroma (this might take a while)...")
        logging.info(f"Creating embeddings with {embedding_model_name} and storing in Chroma...")
        # Use the embeddings object initialized earlier (which now includes the key)
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=vectorstore_path # Save to disk
        )
        vector_store.persist() # Ensure saving
        st.success(f"Vector store created and saved to {vectorstore_path}")
        logging.info(f"Vector store created and saved to {vectorstore_path}")
        st.sidebar.success("Vector store created.")
        return vector_store

    except Exception as e:
        # Catch any other errors during the indexing process
        st.error(f"Error during indexing process: {e}")
        logging.error(f"Error during indexing process: {e}", exc_info=True)
        return None

# This function remains unchanged, primarily used for logging/confirmation
def configure_google_api(api_key: str):
    """Configures the Google Generative AI client."""
    try:
        genai.configure(api_key=api_key)
        logging.info("Google Generative AI client configured successfully.")
        return True
    except Exception as e:
        st.error(f"Failed to configure Google AI client: {e}. Please check your API key.")
        logging.error(f"Failed to configure Google AI client: {e}", exc_info=True)
        return False


# --- Streamlit App ---

def main():
    st.set_page_config(
        page_title="CodeSense",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply Custom CSS (keep your styles here)
    st.markdown("""
        <style>
        /* ... Your CSS ... */
        </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.title("🧠 CodeSense: Java Codebase Q&A")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 1.1em; margin-bottom: 20px;'>
            Ask questions about your Java codebase. Relevant code snippets will be retrieved and used by Gemini to generate an answer.
        </div>
    """, unsafe_allow_html=True)

    # --- Sidebar for Configuration ---
    api_key_configured = False # Flag to track if key is set
    actual_api_key = None      # Variable to hold the validated key

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # --- Google AI API Key Input ---
        google_api_key_input = st.text_input(
            "🔑 Google AI API Key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""), # Pre-fill if env var exists
            help="Get yours from Google AI Studio. Stored securely for this session."
        )

        # Configure Google AI client immediately if key is provided
        if google_api_key_input:
            # Attempt to configure the base library (optional but good for consistency)
            configure_google_api(google_api_key_input)
            api_key_configured = True # Assume configured if user provided input
            actual_api_key = google_api_key_input
        elif "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"]:
             st.info("✅ Using Google AI API Key from environment.")
             # Attempt to configure the base library (optional but good for consistency)
             configure_google_api(os.environ["GOOGLE_API_KEY"])
             api_key_configured = True
             actual_api_key = os.environ["GOOGLE_API_KEY"]

        if not api_key_configured:
            st.warning("⚠️ Please enter your Google AI API Key to proceed.")
            st.stop() # Stop execution if API key is not configured

        # Code Directory Input
        code_directory = st.text_input(
            "📁 Path to Java Codebase",
            value=DEFAULT_CODE_DIR,
            help=f"The root directory containing your .java files. Default: {DEFAULT_CODE_DIR}"
        )
        st.info(f"Using Code Directory: `{os.path.abspath(code_directory)}`")
        st.info(f"Using Vector Store: `{os.path.abspath(VECTORSTORE_DIR)}`")


        # Re-index Button
        force_reindex = st.button("🔄 Re-index Codebase", help="Delete the existing index and rebuild it from the source code directory.")
        if force_reindex:
             st.toast("Re-indexing requested...") # Use toast for temporary messages

    # --- Main Application Logic ---

    # Load or create the vector store, passing the API key <<< FIX
    vector_store = create_or_load_vectorstore(
        code_directory,
        VECTORSTORE_DIR,
        GEMINI_EMBEDDING_MODEL,
        api_key=actual_api_key, # Pass the confirmed key
        force_reindex=force_reindex
    )

    if not vector_store:
        st.warning("⚠️ Vector store could not be loaded or created. Check configuration and logs.")
        st.stop()

    # Initialize LLM (can be cached if model parameters don't change often)
    @st.cache_resource
    def get_llm(model_name, api_key): # <<< Pass API key here too
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=api_key, # <<< Pass API key to LLM init
                convert_system_message_to_human=True
                )
            logging.info(f"Initialized ChatGoogleGenerativeAI with model: {model_name}")
            return llm
        except Exception as e:
            st.error(f"Error initializing Gemini LLM ({model_name}): {e}")
            logging.error(f"Error initializing Gemini LLM ({model_name}): {e}", exc_info=True)
            return None

    # Pass the actual_api_key when getting the LLM <<< FIX
    llm = get_llm(GEMINI_LLM_MODEL, actual_api_key)
    if not llm:
        st.stop()

    # --- Setup RetrievalQA Chain ---
    # Define the prompt template (remains the same)
    prompt_template_str = """You are an expert Java programming assistant. Use the following pieces of context (Java code snippets) to answer the question accurately and concisely.
If you don't know the answer based *only* on the provided context, just say that you cannot answer based on the context provided. Do not make up information.
Always cite the relevant filename(s) from the context if possible.

Context:
{context}

Question: {question}

Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )

    # Create the retriever (remains the same)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Create the RetrievalQA chain (remains the same)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    logging.info("RetrievalQA chain created.")


    # --- Chat Interface (remains the same) ---

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if they exist for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("View Sources Used"):
                    for i, source_doc in enumerate(message["sources"]):
                        try:
                            filename = source_doc.metadata.get('filename', 'Unknown File')
                            page_content = source_doc.page_content
                            st.markdown(f"<div class='source-title'>Source {i+1}: {filename}</div>", unsafe_allow_html=True)
                            st.code(page_content, language="java", line_numbers=False)
                        except Exception as display_err:
                             st.warning(f"Could not display source {i+1}: {display_err}")

    # Accept user input
    if prompt := st.chat_input("Ask about your Java codebase..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using the QA chain
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking... Retrieving context and generating answer..."):
                try:
                    logging.info(f"Invoking QA chain for query: '{prompt[:50]}...'")
                    result = qa_chain.invoke({"query": prompt})

                    answer = result.get("result", "No answer found.")
                    source_documents = result.get("source_documents", [])

                    logging.info(f"Received answer: '{answer[:100]}...'")
                    logging.info(f"Retrieved {len(source_documents)} source documents.")

                    # Display the main answer
                    message_placeholder.markdown(answer)

                    # Add assistant response and sources to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_documents
                    })

                    # Display the sources used in an expander
                    if source_documents:
                        with st.expander("View Sources Used"):
                             for i, source_doc in enumerate(source_documents):
                                try:
                                    filename = source_doc.metadata.get('filename', 'Unknown File')
                                    page_content = source_doc.page_content
                                    st.markdown(f"<div class='source-title'>Source {i+1}: {filename}</div>", unsafe_allow_html=True)
                                    st.code(page_content, language="java", line_numbers=False)
                                except Exception as display_err:
                                    st.warning(f"Could not display source {i+1}: {display_err}")

                except Exception as e:
                    error_message = f"An error occurred while processing your query: {str(e)}"
                    logging.error(f"Error processing query '{prompt[:50]}...': {e}", exc_info=True)
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })

# --- Entry Point (remains the same) ---
if __name__ == "__main__":
    # Create the target directory if it doesn't exist
    if not os.path.exists(DEFAULT_CODE_DIR):
        try:
            os.makedirs(DEFAULT_CODE_DIR)
            logging.warning(f"Default code directory '{DEFAULT_CODE_DIR}' did not exist and was created. Please place your Java files inside.")
        except OSError as e:
            logging.error(f"Failed to create default code directory '{DEFAULT_CODE_DIR}': {e}")

    main()
