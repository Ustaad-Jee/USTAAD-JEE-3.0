import hashlib
import streamlit as st
import fitz  # PyMuPDF
import bleach
import time
from apconfig import AppConfig
from typing import List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantVectorStore
from llm_utils import CustomLLM

# Define constants
KNOWLEDGE_HUB_COLLECTION = "ustaad-jee-knowledge-hub"

def initialize_qdrant() -> bool:
    try:
        api_key = st.secrets.get("QDRANT_API_KEY")
        url = st.secrets.get("QDRANT_URL")
        if not api_key or not url:
            st.error("Qdrant API key or URL not found in secrets")
            return False
        client = QdrantClient(url=url, api_key=api_key)
        st.session_state["qdrant_client"] = client
        collections = client.get_collections()
        existing_collections = [c.name for c in collections.collections]
        if KNOWLEDGE_HUB_COLLECTION not in existing_collections:
            client.create_collection(
                collection_name=KNOWLEDGE_HUB_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"Created collection: {KNOWLEDGE_HUB_COLLECTION}")
            time.sleep(0.5)
        return True
    except Exception as e:
        st.error(f"Qdrant initialization failed: {str(e)}")
        return False

def initialize_embeddings() -> bool:
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state["embeddings"] = embeddings
        Settings.embed_model = embeddings
        return True
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}")
        return False

def initialize_llm() -> bool:
    try:
        if "llm" in st.session_state and st.session_state["llm"]:
            Settings.llm = CustomLLM(st.session_state["llm"])
            return True
        st.error("LLM not initialized. Configure Ustaad Jee's brain first.")
        return False
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        return False

def initialize_components():
    try:
        if not initialize_qdrant():
            return False
        if not initialize_embeddings():
            return False
        if not initialize_llm():
            return False
        return True
    except Exception as e:
        st.error(f"Component initialization failed: {str(e)}")
        return False

def ensure_rag_initialized() -> bool:
    if "rag_initialized" not in st.session_state:
        if initialize_components():
            st.session_state["rag_initialized"] = True
            print("RAG system initialized successfully")
        else:
            st.session_state["rag_initialized"] = False
            print("RAG initialization failed")
    return st.session_state["rag_initialized"]

def parse_document(document: any) -> List[str]:
    try:
        if isinstance(document, str):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            return text_splitter.split_text(document)
        if hasattr(document, 'read'):
            if document.name.endswith(".pdf"):
                with fitz.open(stream=document.read(), filetype="pdf") as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    return text_splitter.split_text(cleaned_text)
            elif document.name.endswith(".txt"):
                text = document.read().decode("utf-8")
                cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                return text_splitter.split_text(cleaned_text)
        raise ValueError("Unsupported document type")
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        raise

def store_base_chunks(text_chunks: List[str], source: str = "knowledge_hub"):
    try:
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]
        vectors = [embeddings.embed_query(chunk) for chunk in text_chunks]
        points = [
            PointStruct(
                id=hashlib.md5(f"{source}_{idx}_{time.time()}".encode()).hexdigest(),
                vector=vector,
                payload={"text": text, "source": source}
            )
            for idx, (text, vector) in enumerate(zip(text_chunks, vectors))
        ]
        client.upsert(
            collection_name=KNOWLEDGE_HUB_COLLECTION,
            points=points,
            wait=True
        )
        print(f"Stored {len(points)} chunks in {KNOWLEDGE_HUB_COLLECTION}")
        return True
    except Exception as e:
        st.error(f"Error storing chunks: {str(e)}")
        return False

def index_document(document_text: str) -> bool:
    try:
        if not document_text.strip():
            st.warning("No document text provided to index")
            return False
        if not ensure_rag_initialized():
            return False
        text_chunks = parse_document(document_text)
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]
        # Initialize LlamaIndex Qdrant vector store
        vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=KNOWLEDGE_HUB_COLLECTION
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = [Document(text=chunk) for chunk in text_chunks]
        # Basic indexing
        store_base_chunks(text_chunks, source="knowledge_hub")
        # Sentence Window indexing
        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        sentence_nodes = sentence_node_parser.get_nodes_from_documents(documents)
        sentence_index = VectorStoreIndex(
            nodes=sentence_nodes,
            storage_context=storage_context
        )
        st.session_state["sentence_window_index"] = sentence_index
        # Auto-merging indexing
        automerging_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        automerging_nodes = automerging_parser.get_nodes_from_documents(documents)
        automerging_index = VectorStoreIndex(
            nodes=automerging_nodes,
            storage_context=storage_context
        )
        st.session_state["automerging_index"] = automerging_index
        st.session_state["vectorstore"] = vector_store
        print("Indexing completed successfully")
        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return False

def index_knowledge_hub_document(document: any) -> bool:
    try:
        text_chunks = parse_document(document)
        document_text = "\n".join(text_chunks) if isinstance(text_chunks, list) else text_chunks
        return index_document(document_text)
    except Exception as e:
        st.error(f"Error indexing knowledge hub document: {str(e)}")
        return False

def clear_indexes():
    try:
        client = st.session_state.get("qdrant_client")
        if client:
            client.delete_collection(KNOWLEDGE_HUB_COLLECTION)
            client.create_collection(
                collection_name=KNOWLEDGE_HUB_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            for key in ["vectorstore", "sentence_window_index", "automerging_index"]:
                if key in st.session_state:
                    del st.session_state[key]
            print(f"Cleared collection: {KNOWLEDGE_HUB_COLLECTION}")
        return True
    except Exception as e:
        st.error(f"Error clearing indexes: {str(e)}")
        return False

def generate_response(query: str, context_text: Optional[str] = None, document_text: Optional[str] = None) -> str:
    try:
        if not query.strip():
            return "Please provide a valid query."
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]
        llm = st.session_state["llm"]
        vector_store = st.session_state.get("vectorstore")
        response = ""
        # Check if vector store and indexes are available
        if vector_store and ("sentence_window_index" in st.session_state or "automerging_index" in st.session_state):
            try:
                # Use sentence window retriever if available
                if "sentence_window_index" in st.session_state:
                    sentence_retriever = st.session_state["sentence_window_index"].as_retriever(
                        similarity_top_k=3
                    )
                    sentence_query_engine = RetrieverQueryEngine(
                        retriever=sentence_retriever,
                        node_postprocessors=[
                            MetadataReplacementPostProcessor(target_metadata_key="window"),
                            SentenceTransformerRerank(top_n=2, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
                        ]
                    )
                    sentence_response = sentence_query_engine.query(query)
                    response = str(sentence_response)
                    print("Used sentence window retriever")
                # Fall back to auto-merging retriever
                elif "automerging_index" in st.session_state:
                    base_retriever = st.session_state["automerging_index"].as_retriever(
                        similarity_top_k=6
                    )
                    retriever = AutoMergingRetriever(
                        base_retriever=base_retriever,
                        vector_store=vector_store,
                        similarity_top_k=3
                    )
                    query_engine = RetrieverQueryEngine.from_args(
                        retriever=retriever,
                        node_postprocessors=[
                            SentenceTransformerRerank(top_n=2, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
                        ]
                    )
                    response = str(query_engine.query(query))
                    print("Used auto-merging retriever")
            except Exception as e:
                st.warning(f"Advanced retrieval failed: {str(e)}. Falling back to basic retrieval.")
                response = ""
        # Basic retrieval if advanced retrieval fails or is not available
        if not response:
            try:
                query_vector = embeddings.embed_query(query)
                search_results = client.search(
                    collection_name=KNOWLEDGE_HUB_COLLECTION,
                    query_vector=query_vector,
                    limit=3,
                    with_payload=True
                )
                retrieved_texts = [hit.payload.get("text", "") for hit in search_results]
                context = "\n".join(retrieved_texts)
                if context_text:
                    context += "\n" + context_text
                if document_text:
                    context += "\n" + document_text
                response = llm.generate(
                    prompt=f"Answer the following question based on the provided context:\n\nQuestion: {query}\n\nContext: {context}",
                    max_tokens=500,
                    temperature=0.7
                )
                print("Used basic retrieval")
            except Exception as e:
                st.error(f"Basic retrieval failed: {str(e)}")
                response = "Error retrieving response. Please try again."
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response."