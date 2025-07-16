import hashlib
import streamlit as st
import fitz  # PyMuPDF
import bleach
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantVectorStore
from llama_index.core import PromptTemplate
from llm_utils import LLMWrapper, LLMProvider, CustomLLM
from typing import List, Optional
from apconfig import AppConfig

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
            return
        if not initialize_embeddings():
            return
        if not initialize_llm():
            return
        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]
        vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=KNOWLEDGE_HUB_COLLECTION
        )
        st.session_state["vectorstore"] = vector_store
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        automerging_parser = HierarchicalNodeParser.from_defaults()
        sentence_window_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embeddings
        )
        automerging_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embeddings
        )
        st.session_state["sentence_window_index"] = sentence_window_index
        st.session_state["automerging_index"] = automerging_index
        st.session_state["node_parser"] = node_parser
        st.session_state["automerging_parser"] = automerging_parser
        st.session_state["rag_initialized"] = True
        print("Components initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.session_state["rag_initialized"] = False

def parse_document(document: any) -> str:
    try:
        if isinstance(document, str):
            return document
        document.seek(0)
        if document.name.endswith(".pdf"):
            doc = fitz.open(stream=document.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if not text.strip():
                raise ValueError("No extractable text in PDF")
            return bleach.clean(text, tags=[], strip=True)
        else:
            text = document.read().decode("utf-8")
            if not text.strip():
                raise ValueError("No extractable text in document")
            return bleach.clean(text, tags=[], strip=True)
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        return ""

def index_document(document_text: str) -> bool:
    try:
        if not document_text.strip():
            st.error("No text provided for indexing")
            return False
        if not st.session_state.get("rag_initialized", False):
            initialize_components()
        documents = [Document(text=document_text)]
        node_parser = st.session_state.get("node_parser")
        automerging_parser = st.session_state.get("automerging_parser")
        sentence_window_index = st.session_state.get("sentence_window_index")
        automerging_index = st.session_state.get("automerging_index")
        if not all([node_parser, automerging_parser, sentence_window_index, automerging_index]):
            st.error("Indexing components not initialized")
            return False
        sentence_nodes = node_parser.get_nodes_from_documents(documents)
        automerging_nodes = automerging_parser.get_nodes_from_documents(documents)
        sentence_window_index.insert_nodes(sentence_nodes)
        automerging_index.insert_nodes(automerging_nodes)
        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return False

def index_knowledge_hub_document(document: any) -> bool:
    try:
        document_text = parse_document(document)
        if not document_text.strip():
            st.error("No extractable text in document")
            return False
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
        if "sentence_window_index" in st.session_state:
            st.session_state["sentence_window_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["vectorstore"],
                embed_model=st.session_state["embeddings"]
            )
        if "automerging_index" in st.session_state:
            st.session_state["automerging_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["vectorstore"],
                embed_model=st.session_state["embeddings"]
            )
        st.session_state.index_version = st.session_state.get("index_version", 0) + 1
    except Exception as e:
        st.error(f"Error clearing indexes: {str(e)}")

@st.cache_data(show_spinner=False, ttl=3600, max_entries=100, hash_funcs={LLMWrapper: lambda x: id(x)})
def retrieve_relevant_chunks(query: str, max_chunks: int = 5) -> List[str]:
    try:
        if not st.session_state.get("rag_initialized", False):
            return []

        automerging_index = st.session_state.get("automerging_index")
        if not automerging_index:
            return []

        # Corrected parameter name: vector_index instead of vector_store_index
        retriever = AutoMergingRetriever(
            vector_index=automerging_index,  # Fixed parameter name
            similarity_top_k=max_chunks,
            postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                SentenceTransformerRerank(top_n=max_chunks)
            ]
        )
        nodes = retriever.retrieve(query)
        return [node.text for node in nodes]
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return []
@st.cache_data(show_spinner=False, ttl=3600, max_entries=100, hash_funcs={LLMWrapper: lambda x: id(x)})
def generate_response(chat_language: str, query: str, context_text: Optional[str], document_text: Optional[str], _glossary_version: int = 0) -> str:
    try:
        if not st.session_state.get("llm"):
            return "Error: LLM not initialized."
        glossary = st.session_state.get("glossary", {})
        glossary_section = "\n".join([
            f"English: {translations.get('English', term)}, Urdu: {translations.get('Urdu', '')}, Roman Urdu: {translations.get('Roman Urdu', '')}"
            for term, translations in glossary.items()
        ]) if glossary else "No glossary terms available."
        last_6_exchanges = "\n".join(
            f"Q: {chat['query']}\nA: {chat['response']}"
            for chat in st.session_state.get("chat_history", [])[-6:]
        ) if st.session_state.get("chat_history") else ""
        context_chunks = retrieve_relevant_chunks(query, max_chunks=5)
        rag_context = "\n".join(context_chunks) if context_chunks else "No relevant chunks found."
        context_section = f"Conversation Context:\n{last_6_exchanges}\nSupplementary Info:\n{context_text or ''}"
        prompt_template = (
            AppConfig.URDU_CHAT_PROMPT if chat_language == "Urdu" else
            AppConfig.ROMAN_URDU_CHAT_PROMPT if chat_language == "Roman Urdu" else
            AppConfig.ENGLISH_CHAT_PROMPT
        )
        prompt = prompt_template.format(
            language=chat_language,
            conversation_history=last_6_exchanges,
            document_text=document_text or "No document provided.",
            rag_context=rag_context,
            supplementary_info=context_text or "No supplementary info provided.",
            glossary_section=glossary_section,
            glossary_translation_rules=AppConfig.GLOSSARY_TRANSLATION_RULES,
            question=query,
            allow_llm_knowledge="False",  # Default to False to restrict to document
            confidence_score="0.9"  # Default confidence score
        )
        response = st.session_state.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        if chat_language == "Urdu":
            response = f"\u200F{response}"
        return response
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        return error_message  # Return error message to be logged in chat history