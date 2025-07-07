#rag_utils
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


# Collection names
BASE_COLLECTION = "ustaad-jee-base"
SENTENCE_WINDOW_COLLECTION = "ustaad-jee-sentence-window"
AUTOMERGING_COLLECTION = "ustaad-jee-automerging"


def initialize_qdrant() -> bool:
    """Initialize Qdrant client and required collections"""
    try:
        api_key = st.secrets.get("QDRANT_API_KEY")
        url = st.secrets.get("QDRANT_URL")
        if not api_key or not url:
            st.error("Qdrant API key or URL not found in secrets")
            return False

        client = QdrantClient(url=url, api_key=api_key)
        st.session_state["qdrant_client"] = client

        # Create required collections if they don't exist
        collections = client.get_collections()
        existing_collections = [c.name for c in collections.collections]

        for collection_name in [BASE_COLLECTION, SENTENCE_WINDOW_COLLECTION, AUTOMERGING_COLLECTION]:
            if collection_name not in existing_collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                print(f"Created collection: {collection_name}")
                time.sleep(1)  # Brief pause between creations

        return True
    except Exception as e:
        st.error(f"Qdrant initialization failed: {str(e)}")
        return False


def initialize_embeddings() -> bool:
    """Initialize HuggingFace embeddings"""
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
    """Initialize LLM for LlamaIndex"""
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
    """Initialize all components for RAG functionality"""
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


def parse_document(document: any) -> List[str]:
    """Parse document into text chunks"""
    try:
        if isinstance(document, str):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            return text_splitter.split_text(document)

        if hasattr(document, 'read'):
            # Handle PDF files
            if document.name.endswith(".pdf"):
                with fitz.open(stream=document.read(), filetype="pdf") as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    return text_splitter.split_text(cleaned_text)

            # Handle text files
            elif document.name.endswith(".txt"):
                text = document.read().decode("utf-8")
                cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                return text_splitter.split_text(cleaned_text)

        raise ValueError("Unsupported document type")
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        raise


def store_base_chunks(text_chunks: List[str]):
    """Store base text chunks in Qdrant"""
    try:
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]

        # Generate embeddings in batch
        vectors = [embeddings.embed_query(chunk) for chunk in text_chunks]

        # Prepare points for upsert
        points = [
            PointStruct(
                id=idx,
                vector=vector,
                payload={"text": text}
            )
            for idx, (text, vector) in enumerate(zip(text_chunks, vectors))
        ]

        # Upsert to Qdrant
        client.upsert(
            collection_name=BASE_COLLECTION,
            points=points,
            wait=True
        )

        print(f"Stored {len(points)} base chunks in Qdrant")
        return True
    except Exception as e:
        st.error(f"Error storing base chunks: {str(e)}")
        return False


def build_sentence_window_index(document_text: str) -> VectorStoreIndex:
    """Build sentence window index for advanced RAG"""
    try:
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]

        # Create vector store
        vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=SENTENCE_WINDOW_COLLECTION
        )

        # Configure node parser
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        # Process document
        document = Document(text=document_text)
        nodes = node_parser.get_nodes_from_documents([document])

        # Create index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embeddings
        )

        return index
    except Exception as e:
        st.error(f"Error building sentence window index: {str(e)}")
        raise


def build_automerging_index(document_text: str) -> VectorStoreIndex:
    """Build auto-merging index for advanced RAG"""
    try:
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]

        # Create vector store
        vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=AUTOMERGING_COLLECTION
        )

        # Configure node parser
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )

        # Process document
        document = Document(text=document_text)
        nodes = node_parser.get_nodes_from_documents([document])
        leaf_nodes = get_leaf_nodes(nodes)

        # Create index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embeddings
        )

        return index
    except Exception as e:
        st.error(f"Error building auto-merging index: {str(e)}")
        raise


def index_document(text_or_file: any) -> bool:
    """Index document with full RAG pipeline"""
    try:
        if not initialize_components():
            return False

        text_chunks = parse_document(text_or_file)
        if not text_chunks:
            st.error("No text chunks created from document.")
            return False

        # Store base chunks
        if not store_base_chunks(text_chunks):
            return False

        # Build advanced indexes
        document_text = "\n".join(text_chunks)
        st.session_state["sentence_window_index"] = build_sentence_window_index(document_text)
        st.session_state["automerging_index"] = build_automerging_index(document_text)
        st.session_state["document_text"] = document_text
        st.session_state["documents_indexed"] = True

        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return False


def clear_indexes():
    """Clear all indexes and vectorstores"""
    try:
        client = st.session_state.get("qdrant_client")
        if client:
            for collection in [BASE_COLLECTION, SENTENCE_WINDOW_COLLECTION, AUTOMERGING_COLLECTION]:
                try:
                    client.delete_collection(collection)
                    print(f"Deleted collection: {collection}")
                except Exception as e:
                    print(f"Error deleting {collection}: {str(e)}")

        # Clear session state
        for key in ["sentence_window_index", "automerging_index", "document_text", "documents_indexed"]:
            if key in st.session_state:
                del st.session_state[key]

        # Clear relevant caches
        st.cache_data.clear()  # Clear all cached data functions
    except Exception as e:
        st.error(f"Error clearing indexes: {str(e)}")


def has_indexed_documents() -> bool:
    """Check if documents are indexed"""
    try:
        if st.session_state.get("documents_indexed"):
            return True

        client = st.session_state.get("qdrant_client")
        if client:
            collection_info = client.get_collection(BASE_COLLECTION)
            return collection_info.points_count > 0

        return False
    except:
        return False


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> Tuple[List[str], float]:
    """Retrieve relevant chunks from Qdrant with confidence score"""
    try:
        if "qdrant_client" not in st.session_state:
            return [], 0.0

        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]

        # Generate query embedding
        query_embedding = embeddings.embed_query(query)

        # Search Qdrant
        results = client.search(
            collection_name=BASE_COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        if not results:
            return [], 0.0

        # Extract text and scores
        chunks = [result.payload["text"] for result in results]
        scores = [result.score for result in results]

        # Calculate average confidence
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        return chunks, avg_confidence
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return [], 0.0


def generate_response(query: str, context_text: Optional[str] = None) -> str:
    try:
        # PRIMARY: Use the user's uploaded document
        document_text = st.session_state.get("uploaded_document", "")
        primary_context = f"DOCUMENT CONTENT:\n{document_text}\n\n" if document_text else ""

        # SECONDARY: Retrieve relevant chunks from Qdrant
        relevant_chunks, confidence = retrieve_relevant_chunks(query)
        rag_context = "\n\n".join(relevant_chunks) if relevant_chunks else ""
        supplementary_context = f"SUPPLEMENTARY INFORMATION:\n{rag_context}\n\n" if rag_context else ""

        # Combine contexts (document first, vectorstore second)
        full_context = primary_context + supplementary_context

        # Additional user-provided context
        if context_text:
            full_context += f"USER CONTEXT:\n{context_text}\n\n"

        # Prepare glossary section
        glossary = st.session_state.get("glossary", {})
        glossary_section = ""
        if isinstance(glossary, dict) and glossary:
            try:
                # Filter valid string key-value pairs
                valid_glossary = {
                    str(eng): str(urdu)
                    for eng, urdu in glossary.items()
                    if isinstance(eng, str) and isinstance(urdu, str) and eng.strip() and urdu.strip()
                }
                if valid_glossary:
                    glossary_section = "\n".join(f"{eng}: {urdu}" for eng, urdu in valid_glossary.items()) + "\n"
            except Exception as e:
                print(f"Glossary formatting error: {str(e)}, Glossary content: {glossary}")
                glossary_section = ""  # Fallback to empty string on error

        # Select and format prompt
        if full_context:
            prompt = AppConfig.RAG_PROMPT.format(
                glossary_section=glossary_section,
                context=full_context,
                question=query
            )
        else:
            prompt = AppConfig.DIRECT_PROMPT.format(
                glossary_section=glossary_section,
                question=query
            )

        # Generate response
        response = st.session_state["llm"].generate(
            prompt=prompt,
            temperature=0.3
        )

        # Handle language translation if needed
        chat_language = st.session_state.get("chat_language", "English")
        if chat_language == "Urdu":
            response = st.session_state["llm"].translate_to_urdu(response)
        elif chat_language == "Roman Urdu":
            response = st.session_state["llm"].translate_to_roman_urdu(response)

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Auto-initialize RAG system
if "rag_initialized" not in st.session_state:
    if initialize_components():
        st.session_state["rag_initialized"] = True
        print("RAG system initialized successfully")
    else:
        st.session_state["rag_initialized"] = False
        print("RAG initialization failed")