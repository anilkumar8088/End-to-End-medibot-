from flask import Flask, render_template, jsonify, request
# --- CHANGE 1: Update Imports ---
# Change from langchain_openai to langchain_google_genai
from langchain_google_genai import ChatGoogleGenerativeAI 
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# --- CHANGE 2: Update API Key Variable Name ---
# Change from OPENAI_API_KEY to GOOGLE_API_KEY
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY') 

# Fix: You must assign the variable, not leave it empty
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Assign the correct variable


embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# --- CHANGE 3 & 4: Initialize the Gemini LLM ---
# 3. Use ChatGoogleGenerativeAI class
# 4. Specify a Gemini model (e.g., "gemini-2.5-flash") and pass the key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.4, 
    max_tokens=500,
    google_api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    
    # Check if the Gemini key is loaded
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found. Check your .env file."
        
    try:
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return f"Sorry, an error occurred while processing your request: {e}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)