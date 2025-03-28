# CodeSense

A powerful Java codebase Q&A system that uses RAG (Retrieval-Augmented Generation) with Google's Gemini model to help developers understand and navigate their Java codebase.

## Features

- 🤖 Powered by Google's Gemini AI model
- 🔍 Intelligent code search and retrieval
- 💬 Interactive chat interface
- 📚 Context-aware responses with code snippets
- 🔄 Easy codebase reindexing
- 🔒 Secure API key management

## Prerequisites

- Python 3.8 or higher
- Google AI API Key (Get it from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Java codebase to analyze

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streamsets-codelense.git
cd streamsets-codelense
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Place your Java codebase in the `my-java-project` directory (or specify a different path in the sidebar)

2. Get your Google AI API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter your Google AI API Key in the sidebar

4. Start asking questions about your Java codebase!

## Project Structure

```
codesense/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── my-java-project/      # Your Java codebase directory
└── java_vectorstore_gemini/  # Vector store for embeddings
```

## How It Works

1. The application indexes your Java codebase using Google's Gemini embeddings
2. When you ask a question, it:
   - Retrieves relevant code snippets
   - Uses Gemini to generate a context-aware response
   - Shows the relevant code snippets with their source files

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini](https://makersuite.google.com/)
- Uses [LangChain](https://www.langchain.com/) for RAG implementation 