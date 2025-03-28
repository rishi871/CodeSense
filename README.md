# 🧠 CodeSense: Java Codebase Q&A Assistant

CodeSense is a web application built with Streamlit that allows you to ask natural language questions about your local Java codebase. It uses a Retrieval-Augmented Generation (RAG) approach, leveraging Google's Gemini models (via LangChain) for embeddings and language generation, and ChromaDB for vector storage.

The application indexes your Java source files, retrieves relevant code snippets based on your query, and then uses the Gemini LLM to synthesize an answer based on the retrieved context.

## ✨ Features

*   🤖 **AI-Powered:** Uses Google's Gemini AI models (`embedding-001` and e.g., `gemini-1.5-pro-latest`).
*   🔍 **Intelligent Retrieval:** Finds relevant code snippets based on semantic similarity to your query.
*   💬 **Interactive Chat UI:** User-friendly interface built with Streamlit.
*   📚 **Context-Aware Answers:** Provides answers grounded in your actual codebase, complete with source code snippets.
*   🔄 **Easy Re-indexing:** Button to refresh the codebase index after code changes.
*   ⚙️ **Flexible Configuration:** Set API keys and codebase paths via the sidebar or a `.env` file.
*   ⚡ **Local Vector Storage:** Uses ChromaDB to store code embeddings locally.
*   🏗️ **LangChain Framework:** Leverages LangChain for robust RAG pipeline orchestration.

## 📋 Prerequisites

1.  **Python:** Version 3.9 - 3.11 installed. **Python 3.10 is recommended** for best compatibility with dependencies. Using a virtual environment is strongly advised. Tools like `pyenv` can help manage multiple Python versions if needed.
2.  **Google AI API Key:** You need an API key from Google AI Studio. Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) to create one.
3.  **Java Codebase:** A local directory containing the Java source code (`.java` files) you want to query.
4.  **Git:** (Optional, for cloning the repository).

## 🚀 Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rishi871/CodeSense.git
    cd CodeSense
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    *   Using Python 3.10 `venv` (replace `python3.10` if your command is different):
        ```bash
        python3.10 -m venv .venv
        ```
    *   Activate the environment:
        *   **macOS / Linux (zsh, bash):**
            ```bash
            source .venv/bin/activate
            ```
        *   **Windows (Command Prompt):**
            ```bash
            .venv\Scripts\activate.bat
            ```
        *   **Windows (PowerShell):**
            ```powershell
            .venv\Scripts\Activate.ps1
            ```
            *(You might need to adjust PowerShell's execution policy: `Set-ExecutionPolicy RemoteSigned -Scope Process`)*

3.  **Install Dependencies:**
    Ensure your virtual environment is activated.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Prepare `.gitignore` (Important!):**
    Before making your first commit (if you cloned and modify), ensure you have a `.gitignore` file in the root directory to prevent committing secrets and unnecessary files. It should contain at least:
    ```gitignore
    # Virtual Environment
    .venv/
    venv/
    env/

    # Vector Store Database
    java_vectorstore_gemini/

    # Python cache
    __pycache__/
    *.pyc
    *.pyo

    # OS generated files
    .DS_Store

    # Secrets / Environment variables
    .env

    # IDE specific files (add as needed)
    .idea/
    ```

## ⚙️ Configuration

1.  **Place Your Java Code:**
    *   Put your Java project files (`.java`) into a directory.
    *   By default, the application looks in `./my-java-project`.
    *   You can change this default path in `app.py` or override it via the sidebar input when running the app.

2.  **Set Google AI API Key:** You have two options:
    *   **Option A: `.env` File (Recommended for local development)**
        Create a file named `.env` in the `CodeSense` root directory:
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
        ```
        Replace `YOUR_GOOGLE_AI_API_KEY` with your actual key. The application will load this automatically.
    *   **Option B: Sidebar Input**
        Run the application and enter your API key directly into the "🔑 Google AI API Key" field in the sidebar. This is stored only for the current session.

## ▶️ Usage

1.  **Ensure your virtual environment is activated.**
2.  **Start the Streamlit application from the `CodeSense` directory:**
    ```bash
    streamlit run app.py
    ```
3.  The application should open in your web browser (usually at `http://localhost:8501`).
4.  **Configure:** If you didn't use a `.env` file, enter your API Key in the sidebar. Verify or update the "Path to Java Codebase".
5.  **Indexing:** On first run (or after clicking "Re-index"), the app will process the Java files. This requires API calls and may take time. The index is saved locally in `java_vectorstore_gemini/`.
6.  **Ask Questions:** Use the chat input to ask about your code!
7.  **Review Sources:** Expand the "View Sources Used" section below the AI's answer to see the code snippets it used.

## 📁 Project Structure

```text
CodeSense/
├── .venv/                  # Virtual environment directory (ignored by git)
├── my-java-project/        # DEFAULT directory for your Java source code (can be changed)
│   └── ... (your .java files)
├── java_vectorstore_gemini/ # Local ChromaDB vector store (ignored by git)
│   └── ... (index files)
├── app.py                  # The main Streamlit application script
├── requirements.txt        # Python dependencies
├── .env                    # Optional: For environment variables (ignored by git)
├── .gitignore              # Specifies intentionally untracked files for Git
└── README.md               # This file


## 🤔 How It Works

1.  **Indexing:** When started (or re-indexed), `app.py` uses `DirectoryLoader` to find `.java` files in the specified codebase directory.
2.  **Chunking:** `RecursiveCharacterTextSplitter` (with Java language awareness) breaks the code into smaller, manageable chunks.
3.  **Embedding:** `GoogleGenerativeAIEmbeddings` converts each code chunk into a numerical vector using the `models/embedding-001` model. API key is required here.
4.  **Storing:** These embeddings and their corresponding text chunks are stored locally in a `Chroma` vector database located in the `java_vectorstore_gemini/` directory.
5.  **Querying (RAG):**
    *   When you ask a question, your query is also embedded using the same model.
    *   `Chroma` searches the database for code chunks with embeddings most similar to your query's embedding.
    *   The `RetrievalQA` chain in LangChain takes your original question and the text of the retrieved code chunks.
    *   A prompt template structures this information for the Gemini LLM (`ChatGoogleGenerativeAI`).
    *   The LLM generates an answer based *only* on the provided code context and question.
    *   The final answer and the source documents (code chunks) are displayed in the Streamlit UI.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please feel free to fork the repository, make changes, and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. (Ensure you add a `LICENSE` file to the repository if you choose this license).

## 🙏 Acknowledgments

*   Built with [Streamlit](https://streamlit.io/)
*   Powered by [Google AI and Gemini Models](https://ai.google.dev/)
*   Uses the [LangChain](https://www.langchain.com/) framework
*   Embeddings stored using [ChromaDB](https://www.trychroma.com/)
