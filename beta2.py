import os
import re
from fastapi import FastAPI, Form, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
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
from transformers import pipeline, AutoTokenizer
from pydantic import BaseModel

os.environ["GOOGLE_API_KEY"] = 'AIzaSyACDfiAkYh8TEqOjXGCWBkFXtlcQ2gmOiY'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

templates = Jinja2Templates(directory="static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2Templates
templates = Jinja2Templates(directory="static")


# %%

# Pydantic model for request body
class ChatRequest(BaseModel):
    query: str

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove content within square brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove content within parentheses
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text.strip()

def preprocess_query(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)    # Remove extra whitespaces
    text = re.sub(r'\[.*?\]', '', text) # Remove content within square brackets
    text = re.sub(r'\(.*?\)', '', text) # Remove content within parentheses
    text = re.sub(r'\d+', '', text)     # Remove digits
    text = re.sub(r'[^\w\s]', '',text)  # Remove special characters and punctuation
    return text.strip()

# Load and process PDFs
def load_pdfs(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
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
def handle_request(query: str):
    if not llm:
        return "Language model initialization failed. Please check the configuration."
    preprocessed_query = preprocess_query(query)
    # if request_type == "question" or "q":
    docs = db.similarity_search(preprocessed_query)
    content = "\n".join(x.page_content for x in docs)
    qa_prompt = ("You are a biblical scholar. Use the following context to answer the user's questions. "
                    "If you don't know the answer, just say that you don't know instead of making up an arbitrary response."
                    "Do not give unnecessarily long answers unless implied in the question."
                    "Give answers in points instead of a single paragraph")
    input_text = f"{qa_prompt}\nContext:\n{content}\nUser Question:\n{query}"
    result = llm.invoke(input_text)
    response = result.content
    # elif request_type == "summarize" or "s":
    # docs = db.similarity_search(query)
    # content = "\n".join(x.page_content for x in docs)
    # summary_prompt = ("You are a biblical scholar. Use the following context to summarize the topic. "
    #                     "Provide a concise summary that captures the main points without losing the essential details."
    #                     "Give answers in points instead of a single paragraph")
    # input_text = f"{summary_prompt}\nContext:\n{content}\nTopic:\n{query}"
    # result = llm.invoke(input_text)
    # response = result.content
    # else:
    #     response = "Invalid request type. Please use 'question' or 'summarize'."
    return response


# %%
# Add a root GET endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define FastAPI routes
# @app.post('/chat')
# async def chat(request: ChatRequest):
#     request_type = request.request_type
#     query = request.query
#     if not request_type or not query:
#         raise HTTPException(status_code=400, detail="Invalid input")
#     response = handle_request(request_type, query)
#     return {"response": response}

@app.post("/submit")
@app.get("/submit")
async def handle_form_submission(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="Invalid input")
    response = handle_request(query)
    return {"message": "Input received", "response": response}


def get_public_ip():
    response = requests.get('https://api.ipify.org')
    return response.text


# Initialize the summarization pipeline and tokenizer
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
except Exception as e:
    raise RuntimeError(f"Failed to load summarization pipeline or tokenizer: {e}")

class SummarizeRequest(BaseModel):
    text: str

def split_text(text, max_tokens=1024):
    tokens = tokenizer.encode(text)
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
    return text_chunks

# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     html_content = """
#         <!DOCTYPE html>
#     <html>
#     <head>
#         <title>Sample Page</title>
#     </head>
#     <body>
#         <h1>Welcome to the Sample Page</h1>
#         <h2> Use the summarize button in this page to get the summarized version of the text.</h2>
#         <p>This is the first sample paragraph. It contains some information that we want to summarize. Summarization is a useful technique in natural language processing to generate concise versions of text.</p>
#         <p>The second paragraph adds more information. It discusses the importance of summarization in various applications such as document summarization, news summarization, and more. Summarizing helps in quickly grasping the essence of large texts.</p>
#         <p>Here is another paragraph with additional details. This text includes information about different techniques used for summarization, including extractive and abstractive summarization methods. Extractive methods select key sentences, while abstractive methods generate new sentences.</p>
#         <p>Let's add some more content to test the summarization. This paragraph contains random information about various topics such as technology, science, and history. The goal is to ensure that the summarization process can handle diverse content efficiently.</p>
#         <p>This paragraph is included to test the summarization of longer texts. Summarization models need to effectively process and condense text without losing essential information. This is especially important for applications like summarizing research papers and legal documents.</p>
#         <p>Finally, this is the last paragraph in the sample page. It reiterates the importance of summarization in today's information-rich world. Effective summarization can save time and provide quick insights into large volumes of text.</p>
#         <button id="summarizeButton">Summarize</button>

#         <script>
#             document.getElementById('summarizeButton').addEventListener('click', function() {
#                 // Collect all paragraphs content
#                 let paragraphs = document.getElementsByTagName('p');
#                 let textContent = Array.from(paragraphs).map(p => p.textContent).join('\n');

#                 // Send the content to backend for summarization
#                 fetch('http://localhost:8000/summarize', {
#                     method: 'POST',
#                     headers: {
#                         'Content-Type': 'application/json'
#                     },
#                     body: JSON.stringify({
#                         text: textContent
#                     })
#                 })
#                 .then(response => response.json())
#                 .then(data => {
#                     alert('Summary: ' + data.summary);
#                 })
#                 .catch(error => {
#                     console.error('Error:', error);
#                 });
#             });
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

@app.post('/summarize', response_model=dict)
async def summarize(request: SummarizeRequest):
    content = request.text
    
    if not content:
        raise HTTPException(status_code=400, detail="The provided content is empty.")

    summaries = []
    try:
        text_chunks = split_text(content, max_tokens=1000)
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize content: {e}")

    combined_summary = ' '.join(summaries)
    
    return {'summary': combined_summary}

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, specify your domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=4000)