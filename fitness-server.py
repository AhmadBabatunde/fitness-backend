from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

store = {}



# Set your Hugging Face API token and Pinecone API key (replace with actual tokens/keys)
huggingfacehub_api_token = os.getenv('huggingfacehub_api_token')
pinecone_api_key = os.getenv('pinecone_api_key')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model_name = "sentence-transformers/all-MiniLM-L6-v2"



# Initialize embeddings
embedding_mod = HuggingFaceEmbeddings(model_name=model_name)

# Initialize Pinecone
vectorstore = PineconeVectorStore(
    index_name="diet-data",
    embedding=embedding_mod,
    pinecone_api_key=pinecone_api_key
)

# Define the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True, 
                             temperature=0.3, api_key= GEMINI_API_KEY,
                             stream=True)

# Define the retriever for history-aware retrieval
retriever = vectorstore.as_retriever(search_kwargs={"top_k": 5})
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt, 
)

# Define the prompt template
prompt_template = """You are an expert in diet and fitness, providing personalized recommendations. Your primary goal is to give precise and actionable advice based on the user's specific context. Use the information provided to tailor your suggestions to their lifestyle, preferences, and health goals. Always reply in English.

    Guidelines:

    - Focus on actionable recommendations: Based on the context, offer concrete suggestions for diet or exercise.
    - Avoid generalities: Tailor each recommendation to the specific details in the context.
    - Be concise yet informative, ensuring the advice is easy to implement.


    Context: {context}

    Chat History: {chat_history}

    Client's Question: {input}

    Response: Make Recommendations based on the client's queries , based on the context provided and the chat history. Additionally, ask reflective questions to encourage deeper exploration. Ensure that your response is concise and not more than 15 line of text.
"""

history_aware_retriever = create_history_aware_retriever(
    llm, vectorstore.as_retriever(), contextualize_q_prompt
)

# Define the prompt template
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "input"]
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize the ConversationalRetrievalChain
qa =  RunnableWithMessageHistory(
rag_chain,
get_session_history,
input_messages_key="input",
history_messages_key="chat_history",
output_messages_key="answer",
)

# Function to generate response
def generate_response(user_input, session_id):
    conversational_qa = RunnableWithMessageHistory(
        qa,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    response = conversational_qa.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    # Remove any long dashes or unwanted characters from the response
    cleaned_response = re.sub(r"^\s*[-–—]+\s*", "", response['answer'])
    cleaned_response = cleaned_response.replace("\n", " ")
    return cleaned_response.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data['question']
    user_id = data.get('user_id', '')
    response = generate_response(question, user_id)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5001)
