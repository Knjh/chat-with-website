import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import pinecone
import os

# Load environment variables
load_dotenv()

# Streamlit App Title
st.title("Chat with a Website")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_ENV = os.getenv("PINECONE_ENV")
#pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Define Pinecone index name
INDEX_NAME = "chat-with-website"

# Step 1: URL Input
url = st.text_input("Enter the URL of the page you want to analyze:")

if url:
    # Step 2: Load the content from the URL
    try:
        st.info("Loading content from the URL...")
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()

        # Step 3: Split the content into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs = text_splitter.split_documents(docs)
        st.success("Content loaded and processed successfully!")
    except Exception as e:
        st.error(f"Error loading content from the URL: {e}")

    # Step 4: Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Step 5: Create a Pinecone vector store
    try:
        st.info("Creating vector store...")
        vectorstore_from_docs = PineconeVectorStore.from_documents(
            docs,
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        retriever = vectorstore_from_docs.as_retriever(search_type='similarity', search_kwargs={'k': 5})

        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

    # Step 6: Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temprature=0.3, max_tokens=600)
    
    # Step 7: Set up the QA chain
    system_prompt = PromptTemplate(
        input_variables=["context"],
        template=(
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you don't know."
            "Use five sentences maximum and keep the answer concise."
            "\n\n{context}"
        )
    )

    prompt = ChatPromptTemplate.from_template(
        "system: {system_prompt}\n Human: {query}"
    )

    question_answer_chain = create_stuff_documents_chain(llm, system_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Step 8: User Query Input
    query = st.text_input("Ask your query about the content:")

    if query:
        st.info("Processing your query...")
        try:
            formatted_prompt = prompt.format(system_prompt=system_prompt.template, query=query)
            response = rag_chain.invoke({"input": formatted_prompt})
            st.success("Query processed successfully!")
            st.write("### Answer:")
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Error processing your query: {e}")
