import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import pinecone
import os
import time

# Load environment variables
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "gleaming-elm"

# Custom CSS
st.set_page_config(page_title="AI Website Chat Assistant", layout="wide")

# Apply custom styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f9f9f9;
    }
    
    /* Cards styling */
    .css-1r6slb0, .css-1y4p8pa {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        background-color: white !important;
    }
    
    /* Headers */
    h1 {
        color: #6C63FF !important;
        font-weight: 600 !important;
    }
    
    h2, h3 {
        color: #4F46E5 !important;
        font-weight: 500 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #6C63FF !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #4F46E5 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 12px;
        background-color: white;
        padding: 1rem;
        margin-bottom: 1rem;
        height: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat messages */
    .user-message {
        background-color: #6C63FF;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        border-bottom-right-radius: 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background-color: #F5F5F5;
        color: #333333;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        border-bottom-left-radius: 5px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 25px !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid #ddd !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        background-color: #fff8e6;
        border-left: 4px solid #ffc107;
    }
    
    .status-success {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    
    .status-error {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("AI Website Chat Assistant")
st.markdown("<p style='color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>Chat with any website content using advanced AI</p>", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I'm your AI assistant. Enter a website URL below, and I'll help you chat about its content."}
    ]

if 'url_processed' not in st.session_state:
    st.session_state.url_processed = False

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if 'website_url' not in st.session_state:
    st.session_state.website_url = ""

# Function to process user query and get response
def process_query(query):
    if not query or not st.session_state.url_processed:
        return
        
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=600)
        
        # Set up the system prompt that includes conversation context
        system_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions about website content.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, say that you don't know.
            Keep your answers conversational and relevant to the question.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Answer the user's question based on the context and chat history.
            """
        )
        
        # Create the QA chain with memory
        question_answer_chain = create_stuff_documents_chain(llm, system_prompt)
        rag_chain = create_retrieval_chain(
            st.session_state.retriever, 
            question_answer_chain
        )
        
        # Get conversation history
        chat_memory = st.session_state.conversation_memory.load_memory_variables({})
        
        # Process the query with conversation history
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_memory.get("chat_history", [])
        })
        
        # Extract the answer
        return response["answer"]
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Create two columns with 1:2 ratio
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Step 1: Enter Website URL")
    
    # URL processing part
    with st.form(key="url_form"):
        url = st.text_input("", placeholder="https://example.com")
        process_url_button = st.form_submit_button("Process Website")
    
    # Process URL when button is clicked
    if process_url_button and url:
        try:
            with st.spinner("Loading content from the URL..."):
                # Add a status message
                st.markdown("<div class='status-indicator status-processing'>🔄 Processing website content...</div>", unsafe_allow_html=True)
                
                # Load the content from the URL
                loader = UnstructuredURLLoader(urls=[url])
                docs = loader.load()
                
                # Split the content into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                docs = text_splitter.split_documents(docs)
                
                # Create embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
                # Create a Pinecone vector store
                vectorstore_from_docs = PineconeVectorStore.from_documents(
                    docs,
                    index_name=INDEX_NAME,
                    embedding=embeddings
                )
                
                # Store the retriever in session state
                st.session_state.retriever = vectorstore_from_docs.as_retriever(search_type='similarity', search_kwargs={'k': 5})
                st.session_state.url_processed = True
                st.session_state.website_url = url
                
                # Reset conversation memory for the new website
                st.session_state.conversation_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                # Reset chat history but keep the welcome message
                st.session_state.chat_history = [
                    {"role": "assistant", "content": f"I've analyzed the content from {url}. What would you like to know about it?"}
                ]
                
                # Success message
                st.markdown("<div class='status-indicator status-success'>✅ Website processed successfully!</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"<div class='status-indicator status-error'>❌ Error: {str(e)}</div>", unsafe_allow_html=True)
    
    # Helper text
    st.markdown("<p style='color: #666; font-size: 0.9rem; margin-top: 1rem;'>Enter the URL of any website you want to chat about. Our AI will analyze the content and prepare to answer your questions.</p>", unsafe_allow_html=True)
    
    # Show current website being processed
    if st.session_state.website_url:
        st.markdown(f"<p style='color: #4F46E5; font-size: 0.9rem; margin-top: 1rem;'><strong>Currently chatting about:</strong> {st.session_state.website_url}</p>", unsafe_allow_html=True)

    # Add a button to clear the conversation
    if st.session_state.url_processed:
        if st.button("Clear Conversation"):
            # Keep the website URL and retriever but reset the conversation
            st.session_state.chat_history = [
                {"role": "assistant", "content": f"I've analyzed the content from {st.session_state.website_url}. What would you like to know about it?"}
            ]
            # Reset conversation memory
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            st.rerun()

with col2:
    st.markdown("### Chat")
    
    # Display chat container with fixed height and scrolling
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Message input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("", placeholder="Ask a question about the website...", 
                                 disabled=not st.session_state.url_processed)
        submit_button = st.form_submit_button("Send")
    
    # Process the query when form is submitted
    if submit_button and user_query and st.session_state.url_processed:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Add the user message to conversation memory
        st.session_state.conversation_memory.chat_memory.add_user_message(user_query)
        
        # Process the query
        with st.spinner("Thinking..."):
            answer = process_query(user_query)
            
            # Add AI response to conversation memory
            st.session_state.conversation_memory.chat_memory.add_ai_message(answer)
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Rerun to update the UI with the new messages
        st.rerun()

# Footer
st.markdown("<div style='text-align: center; color: #666; margin-top: 2rem; font-size: 0.9rem;'>Powered by Gemini AI © 2025</div>", unsafe_allow_html=True)
