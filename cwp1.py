# %%
import os
import re
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel
import requests

os.environ["GOOGLE_API_KEY"] = 'AIzaSyACDfiAkYh8TEqOjXGCWBkFXtlcQ2gmOiY'
app = FastAPI()


# %%

# Pydantic model for request body
class ChatRequest(BaseModel):
    request_type: str
    query: str

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove content within square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove content within parentheses
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text.strip()

# Load and process PDFs
def load_pdfs(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            pdf_texts.extend(texts)
    return pdf_texts

# Initialize models and database
def initialize_models_and_db(pdf_directory):
    pdf_texts = load_pdfs(pdf_directory)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(pdf_texts, embeddings)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    except Exception as e:
        print(f'Failed to initialize ChatGoogleGenerativeAI: {e}')
        llm = None
    return db, llm


# %%

pdf_directory = r'C:\Users\abiram\Desktop\Test1\Folder2\pdfs'
db, llm = initialize_models_and_db(pdf_directory)

# Handle incoming requests
def handle_request(request_type, query):
    if not llm:
        return "Language model initialization failed. Please check the configuration."

    if request_type == "question":
        docs = db.similarity_search(query)
        content = "\n".join(x.page_content for x in docs)
        qa_prompt = ("You are a biblical scholar. Use the following context to answer the user's questions. "
                     "If you don't know the answer, just say that you don't know instead of making up an arbitrary response.")
        input_text = f"{qa_prompt}\nContext:\n{content}\nUser Question:\n{query}"
        result = llm.invoke(input_text)
        response = result.content
    elif request_type == "summarize":
        docs = db.similarity_search(query)
        content = "\n".join(x.page_content for x in docs)
        summary_prompt = ("You are a biblical scholar. Use the following context to summarize the topic. "
                          "Provide a concise summary that captures the main points without losing the essential details.")
        input_text = f"{summary_prompt}\nContext:\n{content}\nTopic:\n{query}"
        result = llm.invoke(input_text)
        response = result.content
    else:
        response = "Invalid request type. Please use 'question' or 'summarize'."
    return response


# %%
# Add a root GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Define FastAPI routes
@app.post('/chat')
async def chat(request: ChatRequest):
    request_type = request.request_type
    query = request.query
    if not request_type or not query:
        raise HTTPException(status_code=400, detail="Invalid input")
    response = handle_request(request_type, query)
    return {"response": response}

def get_public_ip():
    response = requests.get('https://api.ipify.org')
    return response.text


# %%
if __name__ == '__main__':
    import uvicorn
    import asyncio

    public_ip = get_public_ip()
    print(f'Public IP: {public_ip}')

    async def run_server():
        config = uvicorn.Config(app=app, host='0.0.0.0', port=5000)
        server = uvicorn.Server(config)
        await server.serve()

    # Check if there's already a running event loop
    if asyncio.get_event_loop().is_running():
        # Create a task in the current event loop
        asyncio.ensure_future(run_server())
    else:
        # If no event loop is running, use asyncio.run
        asyncio.run(run_server())


# %%



