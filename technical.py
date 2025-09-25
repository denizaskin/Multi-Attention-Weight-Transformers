import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()

def create_rag_system():
    # Load the document
    loader = TextLoader("your_text_file.txt")
    documents = loader.load()
    
    # Set path for vector store
    vector_store_path = "vector_store"
    
    # Try to load existing vector store
    if os.path.exists(vector_store_path):
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True  # Only use if you trust the source
        )
    else:
        print("Creating new vector store...")
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Save the vector store locally
        vectorstore.save_local(vector_store_path)
    
    # Create a question-answering chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain

def main():
    # Create the RAG system
    qa_chain = create_rag_system()
    
    # Example query
    query = "What era did Napoleon Bonaparte live in?"
    result = qa_chain.invoke(query)
    print(f"Question: {query}")
    print(f"Answer: {result}")

if __name__ == "__main__":
    main()