import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory # Import Flask components
import logging # Use logging for better output

# --- Langchain Imports ---
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
# Use FAISS from core langchain_community if available, or adjust import as needed
from langchain_community.vectorstores import FAISS
# Use HuggingFaceEmbeddings from langchain_community or langchain_huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings # Or from langchain_huggingface
from langchain.chains import RetrievalQA
# Adjust ChatOpenAI import based on your langchain version
# from langchain_community.chat_models import ChatOpenAI # Older versions
from langchain_openai import ChatOpenAI # Newer versions >= 0.2.0

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Check for necessary environment variables
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    logging.error("❌ OPENROUTER_API_KEY not found in environment variables. Please set it.")
    exit() # Exit if the key is essential and missing

api_base = "https://openrouter.ai/api/v1"
reference_folder = "reference_files"
chat_history_file = os.path.join(reference_folder, "chat_history.txt")
# Consider making the model name an environment variable too
# Ensure this model is available on OpenRouter and compatible with your key/tier
llm_model_name = os.getenv("OPENROUTER_MODEL_NAME", "meta-llama/llama-4-maverick:free")

# Ensure reference_files directory exists
os.makedirs(reference_folder, exist_ok=True)

# --- Global Variables ---
# Initialize these globally so they are set up once
qa_chain = None
initialization_error = None

# --- Helper Functions ---
def load_documents(folder_path):
    """Loads documents from .txt and .pdf files in the specified folder."""
    documents = []
    if not os.path.isdir(folder_path):
        logging.warning(f"⚠️ Reference folder '{folder_path}' not found.")
        return documents # Return empty list if folder doesn't exist

    logging.info(f"📂 Loading documents from: {folder_path}")
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        loader = None
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(filepath, encoding="utf-8")
            except Exception as e: # Catch potential encoding issues during init
                 logging.error(f"❌ Error initializing TextLoader for {filename} (maybe wrong encoding?): {e}")
                 continue
        elif filename.endswith(".pdf"):
            loader = PyMuPDFLoader(filepath)
        else:
            logging.debug(f"Skipping non .txt/.pdf file: {filename}")
            continue # Skip unknown files

        if loader:
            try:
                loaded_docs = loader.load()
                if loaded_docs: # Check if loader actually returned documents
                     documents.extend(loaded_docs)
                     logging.info(f"📄 Loaded {filename} ({len(loaded_docs)} parts)")
                else:
                     logging.warning(f"⚠️ Loader for {filename} returned no documents.")
            except Exception as e:
                logging.error(f"❌ Error loading {filename}: {e}")
    logging.info(f"📚 Total document parts loaded: {len(documents)}")
    return documents

# --- Initialization Function ---
def initialize_qa_system():
    """Loads data, creates embeddings, vector store, and the QA chain."""
    global qa_chain, initialization_error
    try:
        docs = load_documents(reference_folder)
        if not docs:
            raise ValueError(f"No documents loaded from '{reference_folder}'. Cannot initialize QA system.")

        logging.info("🧠 Creating embeddings (this might take a while)...")
        # You might want to specify a model for HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Example
        embeddings = HuggingFaceEmbeddings()

        logging.info("💾 Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()
        logging.info("✅ Vector store created successfully.")

        logging.info(f"🤖 Initializing LLM ({llm_model_name}) via OpenRouter...")
        llm = ChatOpenAI(
            openai_api_key=openrouter_key,
            openai_api_base=api_base,
            model_name=llm_model_name,
            # Add other parameters like temperature if needed
            # temperature=0.7
        )

        # Use RetrievalQA - simple chain good for direct Q&A on docs
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Common chain type, adjust if needed
            retriever=retriever,
            return_source_documents=False # Set to False because we only need the answer string
        )
        logging.info("✅ QA Chain ready!")

    except Exception as e:
        logging.exception("❌❌❌ CRITICAL ERROR during QA system initialization.")
        initialization_error = str(e)
        qa_chain = None # Ensure qa_chain is None if setup fails

# --- Flask App Setup ---
app = Flask(__name__)

# Run initialization *before* the first request
# Using @app.before_first_request is deprecated in newer Flask versions.
# It's better to initialize eagerly when the module is loaded.
initialize_qa_system()


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML file."""
    # Assumes qa_ui.html is in the same directory as the python script
    logging.info("Serving index page (qa_ui.html)")
    try:
        return send_from_directory('.', 'qa_ui.html')
    except FileNotFoundError:
         logging.error("❌ qa_ui.html not found in the current directory!")
         return "Error: UI file not found.", 404

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles POST requests to the /ask endpoint for answering questions."""
    logging.debug("Received request for /ask")

    # Check if QA system initialized correctly
    if initialization_error:
         logging.error(f"Returning initialization error: {initialization_error}")
         return jsonify({"error": f"QA system failed to initialize: {initialization_error}"}), 503 # 503 Service Unavailable

    if qa_chain is None: # Should be caught by initialization_error, but double-check
        logging.error("QA chain is None, though no initialization error was recorded.")
        return jsonify({"error": "QA system is not available."}), 503

    # Check if request is JSON
    if not request.is_json:
        logging.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        logging.warning("Missing 'query' in request JSON")
        return jsonify({"error": "Missing 'query' in request body"}), 400

    logging.info(f"🙋 Received query: {query}")

    try:
        # --- Get the answer from the QA chain ---
        # Use invoke for newer Langchain versions, run for older. Adjust as needed.
        # result = qa_chain.run(query) # Older Langchain
        response = qa_chain.invoke({"query": query}) # Newer Langchain >= 0.1.0

        # Check the structure of the response from invoke
        if isinstance(response, dict) and 'result' in response:
             result_text = response['result']
        elif isinstance(response, str): # Handle if .run() was used or invoke returns string
             result_text = response
        else:
             logging.error(f"Unexpected response format from QA chain: {type(response)}")
             return jsonify({"error": "Received unexpected response format from QA system."}), 500


        if result_text is None:
             logging.warning("QA chain returned None or empty result.")
             # Decide how to handle None results - maybe the LLM couldn't answer
             # Send a specific message or an error? Let's send a specific message.
             result_text = "I couldn't find an answer based on the provided documents."
             # return jsonify({"error": "Failed to get an answer from the QA system."}), 500

        logging.info(f"🤖 Sending answer: {result_text[:100]}...") # Log beginning of answer

        # Log chat to file (append mode)
        try:
            with open(chat_history_file, "a", encoding="utf-8") as file:
                file.write(f"You: {query}\n")
                file.write(f"Model: {result_text}\n")
        except Exception as log_e:
            logging.warning(f"⚠️ Could not write to chat history file '{chat_history_file}': {log_e}")

        # Return the answer
        return jsonify({"answer": result_text})

    except Exception as e:
        logging.exception("❌ Error processing query in /ask endpoint.") # Log full traceback
        return jsonify({"error": f"An internal error occurred while processing the question: {e}"}), 500

# --- Run Flask App ---
if __name__ == "__main__":
    # host='0.0.0.0' makes the server accessible from other devices on your network
    # debug=True enables auto-reloading on code changes and detailed error pages (disable for production!)
    logging.info("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=False) # Set debug=False for production