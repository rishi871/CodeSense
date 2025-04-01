# app.py

import streamlit as st
import os
import shutil
import logging
from operator import itemgetter  # For LCEL

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# --- MODIFICATION: LCEL Imports ---
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
# --- End LCEL Imports ---
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- Configuration ---
# (Keep your configuration the same)
DEFAULT_CODE_DIR = "./my-java-project"
VECTORSTORE_DIR = "./java_vectorstore_gemini"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
<<<<<<< HEAD
GEMINI_LLM_MODEL = "gemini-2.5-pro-exp-03-25"
=======
GEMINI_LLM_MODEL = "gemini-2.5-pro-exp-03-25" # Or "gemini-1.5-flash-latest", "gemini-pro"
>>>>>>> origin/main

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LangChain & Indexing Functions ---
# (create_or_load_vectorstore remains the same)
@st.cache_resource(show_spinner="Loading and Indexing Java Code (Gemini)...")
def create_or_load_vectorstore(
    code_dir: str,
    vectorstore_path: str,
    embedding_model_name: str,
    api_key: str,
    force_reindex: bool = False
) -> Chroma | None:
    # ... (function code is unchanged) ...
    vector_store = None
    embeddings = None

    if not api_key:
        st.error("API Key is missing. Cannot initialize embeddings.")
        logging.error("API Key is missing in create_or_load_vectorstore.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_key
        )
        logging.info(f"Initialized GoogleGenerativeAIEmbeddings with model: {embedding_model_name}")
    except Exception as e:
        st.error(f"Failed to initialize Google Embeddings. Error: {e}. Make sure your API key is valid and has permissions.")
        logging.error(f"Failed to initialize Google Embeddings: {e}", exc_info=True)
        return None

    if os.path.exists(vectorstore_path) and not force_reindex:
        try:
            st.info(f"Loading existing vector store from {vectorstore_path}")
            logging.info(f"Loading existing vector store from {vectorstore_path}")
            vector_store = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            st.sidebar.success("Vector store loaded.")
            logging.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            st.warning(f"Failed to load existing vector store: {e}. Will attempt to re-index.")
            logging.warning(f"Failed to load existing vector store from {vectorstore_path}: {e}", exc_info=True)
            try:
                shutil.rmtree(vectorstore_path)
                logging.info(f"Removed potentially corrupted directory: {vectorstore_path}")
            except Exception as rm_err:
                st.error(f"Error removing corrupted vector store directory: {rm_err}")
                logging.error(f"Error removing corrupted directory {vectorstore_path}: {rm_err}", exc_info=True)
                return None

    elif os.path.exists(vectorstore_path) and force_reindex:
         st.info(f"Re-indexing: Removing existing vector store at {vectorstore_path}")
         logging.info(f"Re-indexing: Removing existing vector store at {vectorstore_path}")
         try:
            shutil.rmtree(vectorstore_path)
         except Exception as e:
            st.error(f"Error removing vector store directory for re-indexing: {e}")
            logging.error(f"Error removing vector store directory {vectorstore_path} for re-indexing: {e}", exc_info=True)
            return None

    if not os.path.exists(code_dir):
        st.error(f"Error: Code directory '{code_dir}' not found.")
        logging.error(f"Code directory '{code_dir}' not found.")
        return None

    st.info(f"Creating new vector store from code in {code_dir}")
    logging.info(f"Creating new vector store from code in {code_dir}")
    try:
        loader = DirectoryLoader(
            code_dir,
            glob="**/*.java",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} raw documents.")

        if not documents:
            st.warning("No .java files found in the specified directory. The index will be empty.")
            logging.warning(f"No .java files found in {code_dir}.")
            return None

        java_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=1500, chunk_overlap=150
        )
        split_docs = java_splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_docs)} chunks.")

        for doc in split_docs:
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

        st.info(f"Creating embeddings using Google AI '{embedding_model_name}' and storing in Chroma (this might take a while)...")
        logging.info(f"Creating embeddings with {embedding_model_name} and storing in Chroma...")
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=vectorstore_path
        )
        vector_store.persist()
        st.success(f"Vector store created and saved to {vectorstore_path}")
        logging.info(f"Vector store created and saved to {vectorstore_path}")
        st.sidebar.success("Vector store created.")
        return vector_store

    except Exception as e:
        st.error(f"Error during indexing process: {e}")
        logging.error(f"Error during indexing process: {e}", exc_info=True)
        return None


# (configure_google_api remains the same)
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

# Helper function to format documents
def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into a single string for context."""
    return "\n\n".join([f"--- Source: {doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))} ---\n{doc.page_content}" for doc in docs])

# --- Streamlit App ---

def main():
    st.set_page_config(
        page_title="CodeSense",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # (CSS remains the same)
    st.markdown("""
        <style>
        .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; border: 1px solid #eee; }
        .stChatMessage:nth-child(odd) { background-color: #e1f5fe; } /* User */
        .stChatMessage:nth-child(even) { background-color: #f1f8e9; } /* Assistant */
        .stExpander { border: 1px solid #ddd; border-radius: 8px; margin-top: 5px;}
        .stCodeBlock { border-radius: 5px; }
        .source-title { font-weight: bold; color: #424242; margin-bottom: 5px; font-size: 0.9em;}
        .main > div:first-child > div:first-child > div:first-child > div:first-child { text-align: center; }
        .main .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

    # (Header remains the same)
    st.title("🧠 CodeSense: Java Codebase Q&A")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 1.1em; margin-bottom: 20px;'>
            Ask questions about your Java codebase. Relevant code snippets will be retrieved and used by Gemini to generate an answer, remembering the conversation context.
        </div>
    """, unsafe_allow_html=True)

    # (Sidebar and API Key logic remains the same)
    api_key_configured = False
    actual_api_key = None
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        google_api_key_input = st.text_input(
            "🔑 Google AI API Key",
            type="password",
            value=st.session_state.get("google_api_key", os.environ.get("GOOGLE_API_KEY", "")),
            help="Get yours from Google AI Studio. Stored securely for this session."
        )
        if google_api_key_input:
             st.session_state.google_api_key = google_api_key_input

        if "google_api_key" in st.session_state and st.session_state.google_api_key:
            actual_api_key = st.session_state.google_api_key
            if configure_google_api(actual_api_key):
                api_key_configured = True
            else:
                 st.session_state.google_api_key = None
                 actual_api_key = None
        elif "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"]:
             if configure_google_api(os.environ["GOOGLE_API_KEY"]):
                st.info("✅ Using Google AI API Key from environment variable.")
                api_key_configured = True
                actual_api_key = os.environ["GOOGLE_API_KEY"]
                st.session_state.google_api_key = actual_api_key

        if not api_key_configured:
            st.warning("⚠️ Please enter a valid Google AI API Key to proceed.")
            st.stop()

        code_directory = st.text_input(
            "📁 Path to Java Codebase",
            value=DEFAULT_CODE_DIR,
            help=f"The root directory containing your .java files. Default: {DEFAULT_CODE_DIR}"
        )
        st.info(f"Using Code Directory: `{os.path.abspath(code_directory)}`")
        st.info(f"Using Vector Store: `{os.path.abspath(VECTORSTORE_DIR)}`")

        force_reindex = st.button("🔄 Re-index Codebase", help="Delete the existing index and rebuild it from the source code directory.")
        if force_reindex:
             st.toast("Re-indexing requested...")
             st.cache_resource.clear()
             # Delete memory and messages from session state before rerun
             if "memory" in st.session_state: del st.session_state.memory
             if "messages" in st.session_state: del st.session_state.messages
             st.rerun() # Rerun the script

    # --- Main Application Logic ---

    # === FORCE EARLY MEMORY INITIALIZATION ===
    # This ensures 'st.session_state.memory' always exists, even if empty
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )
        logging.info("Forced initial creation of ConversationBufferMemory.")
    # === END FORCED INITIALIZATION ===


    vector_store = create_or_load_vectorstore(
        code_directory,
        VECTORSTORE_DIR,
        GEMINI_EMBEDDING_MODEL,
        api_key=actual_api_key,
        force_reindex=force_reindex
    )

    if not vector_store:
        st.error("⚠️ Vector store could not be loaded or created. Please check path and logs.")
        st.stop()

    # (LLM initialization remains the same)
    @st.cache_resource
    def get_llm(model_name, api_key):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=api_key,
                convert_system_message_to_human=True
                )
            logging.info(f"Initialized ChatGoogleGenerativeAI with model: {model_name}")
            return llm
        except Exception as e:
            st.error(f"Error initializing Gemini LLM ({model_name}): {e}")
            logging.error(f"Error initializing Gemini LLM ({model_name}): {e}", exc_info=True)
            return None

    llm = get_llm(GEMINI_LLM_MODEL, actual_api_key)
    if not llm:
        st.error("LLM could not be initialized. Cannot proceed.")
        st.stop()

    # --- MODIFICATION: Load Memory Before Chain and Modify LCEL ---

    # Prompt template remains the same
    prompt_template_str = """You are an expert Java programming assistant. Use the following pieces of context (Java code snippets) and the chat history to answer the question accurately and concisely.
If you don't know the answer based *only* on the provided context and chat history, just say that you cannot answer based on the information provided. Do not make up information.
Always cite the relevant filename(s) from the context if possible using the source metadata provided in the context section.

Chat History:
{chat_history}

Retrieved Context (Code Snippets):
{context}

Question: {question}

Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template_str,
        input_variables=["chat_history", "context", "question"]
    )

    # Initialize chat history for display (if needed)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logging.info("Initialized new chat session state for display.")

    # Now the conditional load/load history.
    # Note: The forced initialization above ensures memory exists,
    # but we might need to populate it from existing messages if they exist.
    if "memory" in st.session_state and not st.session_state.memory.load_memory_variables({})['chat_history']:
        # Check if memory exists BUT is empty (e.g., after forced init)
        # AND if messages exist from a previous session state
        if "messages" in st.session_state and st.session_state.messages:
            logging.info("Memory exists but is empty. Attempting to load from st.session_state.messages...")
            temp_memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    temp_memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    ai_content = msg.get("content", "")
                    if ai_content: temp_memory.chat_memory.add_ai_message(ai_content)
            st.session_state.memory = temp_memory # Replace empty memory with loaded one
            logging.info(f"Loaded {len(st.session_state.messages)} messages into memory.")


    # === LOAD MEMORY VARIABLES BEFORE CHAIN ===
    # Ensure memory exists before loading (should always be true now)
    if "memory" in st.session_state:
        memory_vars = st.session_state.memory.load_memory_variables({})
        logging.info(f"Loaded memory variables: {memory_vars}")
    else:
        # This case should ideally not happen due to forced init, but added defensively
        memory_vars = {"chat_history": ""} # Default to empty history
        logging.warning("Memory object not found before loading vars, defaulting to empty history.")


    # --- Define Retriever and LCEL Chain (NOW SAFE TO ACCESS MEMORY) ---

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Define the LCEL Chain
    rag_chain = (
        RunnableParallel(
            retrieved_docs=itemgetter("question") | retriever,
            # Pass loaded memory_vars, no more direct access to st.session_state
            memory_load_result=RunnablePassthrough.assign(chat_history=lambda x: memory_vars.get('chat_history', "")),
            question=itemgetter("question")
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x['retrieved_docs']),
            # Pass chat_history through the 'memory_load_result'
            chat_history=lambda x: x['memory_load_result']['chat_history']
        )
        | {
             "context": itemgetter("context"),
             "chat_history": itemgetter("chat_history"),
             "question": itemgetter("question"),
             "retrieved_docs": itemgetter("retrieved_docs")
          }
        | RunnablePassthrough.assign(
            answer = {
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
                "question": itemgetter("question"),
            } | QA_PROMPT | llm | StrOutputParser()
        )
    )
    logging.info("LCEL RAG chain created.")

    # --- Chat Interface ---

    # (Display chat messages from history - **CORRECTED**)
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Check if sources exist in *this* specific message
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("View Sources Used"):
                        # **FIX:** Iterate over message["sources"], not source_documents
                        for i, source_doc in enumerate(message["sources"]):
                            try:
                                filename = source_doc.metadata.get('filename', source_doc.metadata.get('source', 'Unknown File'))
                                page_content = source_doc.page_content
                                st.markdown(f"<div class='source-title'>Source {i+1}: {filename}</div>", unsafe_allow_html=True)
                                st.code(page_content, language="java", line_numbers=False)
                            except Exception as display_err:
                                 st.warning(f"Could not display source {i+1}: {display_err}")

    # Accept user input
    if prompt := st.chat_input("Ask about your Java codebase..."):
        # Ensure messages list exists if it was deleted during reindex/restart
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking... Retrieving context and generating answer..."):
                try:
                    logging.info(f"Invoking LCEL chain for question: '{prompt[:50]}...'")

                    # Invoke the LCEL chain
                    result = rag_chain.invoke({"question": prompt})

                    answer = result.get("answer", "Sorry, I could not generate an answer.")
                    # This is the correct place to get source_documents for the *current* answer
                    source_documents = result.get("retrieved_docs", [])

                    logging.info(f"Received answer: '{answer[:100]}...'")
                    logging.info(f"Retrieved {len(source_documents)} source documents.")

                    # Manual Memory Update (Check if memory exists, although it should now)
                    if "memory" in st.session_state:
                        if prompt and answer:
                            st.session_state.memory.save_context(
                               {"question": prompt},
                               {"output": answer}
                            )
                            logging.info("Manually saved context (question/output) to memory.")
                        else:
                            logging.warning("Skipping memory save due to missing prompt or answer.")
                    else:
                        logging.error("Memory object not found in session state during save context attempt.")


                    message_placeholder.markdown(answer)

                    # Add assistant response and sources to display history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_documents # Store the sources for *this* response
                    })

                    # Display sources for the *current* response
                    if source_documents:
                        with st.expander("View Sources Used"):
                             for i, source_doc in enumerate(source_documents):
                                try:
                                    filename = source_doc.metadata.get('filename', source_doc.metadata.get('source', 'Unknown File'))
                                    page_content = source_doc.page_content
                                    st.markdown(f"<div class='source-title'>Source {i+1}: {filename}</div>", unsafe_allow_html=True)
                                    st.code(page_content, language="java", line_numbers=False)
                                except Exception as display_err:
                                    st.warning(f"Could not display source {i+1}: {display_err}")

                except Exception as e:
                    logging.error(f"Error processing question '{prompt[:50]}...': {e}", exc_info=True)
                    error_message = f"An error occurred: {str(e)}"
                    message_placeholder.error(error_message)
                    # Ensure messages list exists before appending error
                    if "messages" not in st.session_state:
                         st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })

# --- Entry Point ---
# (Entry point logic remains the same)
if __name__ == "__main__":
    if not os.path.exists(DEFAULT_CODE_DIR):
        try:
            os.makedirs(DEFAULT_CODE_DIR)
            logging.warning(f"Default code directory '{DEFAULT_CODE_DIR}' did not exist and was created. Please place your Java source files inside.")
            placeholder_path = os.path.join(DEFAULT_CODE_DIR, "placeholder.java")
            if not os.path.exists(placeholder_path):
                with open(placeholder_path, "w") as f:
                    f.write("// Please add your Java source files to this directory\n")
                    f.write("public class Placeholder {}")
                logging.info(f"Created placeholder file: {placeholder_path}")
        except OSError as e:
            logging.error(f"Failed to create default code directory '{DEFAULT_CODE_DIR}': {e}")

    main()
