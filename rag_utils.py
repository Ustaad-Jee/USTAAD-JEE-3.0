import streamlit as st
import pymupdf4llm
import bleach
import time
from apconfig import AppConfig
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index_vector_stores_qdrant import QdrantVectorStore as LlamaQdrantVectorStore
from llm_utils import CustomLLM
import traceback

COLLECTION_NAME = "ustaad-jee-textbooks"

def initialize_qdrant() -> bool:
    """Initialize Qdrant client"""
    try:
        api_key = st.secrets.get("QDRANT_API_KEY")
        url = st.secrets.get("QDRANT_URL")
        print(f"Qdrant API Key: {api_key[:5]}..." if api_key else "No API key found")  # Debug
        print(f"Qdrant URL: {url}")  # Debug
        if not api_key or not url:
            st.error("Qdrant API key or URL not found in secrets")
            return False

        client = QdrantClient(url=url, api_key=api_key)
        st.session_state["qdrant_client"] = client

        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"Qdrant collections: {collection_names}")  # Debug

        if COLLECTION_NAME not in collection_names:
            print(f"Creating Qdrant collection: {COLLECTION_NAME}")  # Debug
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        print("Qdrant initialized successfully")  # Debug
        return True
    except Exception as e:
        st.error(f"Qdrant initialization failed: {str(e)}")
        print(f"Qdrant error details: {str(e)}")  # Debug
        return False

def initialize_embeddings() -> bool:
    """Initialize HuggingFace embeddings"""
    try:
        print("Initializing embeddings")  # Debug
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state["embeddings"] = embeddings
        st.session_state["embedding_provider"] = "HuggingFace"
        print(f"Embeddings initialized: {type(embeddings)}")  # Debug
        return True
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}")
        print(f"Embeddings error details: {str(e)}")  # Debug
        return False

def initialize_llm() -> bool:
    """Initialize LLM for LlamaIndex"""
    try:
        if "llm" in st.session_state and st.session_state["llm"]:
            Settings.llm = CustomLLM(st.session_state["llm"])
            print(f"LLM initialized: {type(Settings.llm)}")  # Debug
            return True
        else:
            st.error("LLM not initialized in session state. Please configure Ustaad Jee's brain first.")
            print("LLM initialization failed: No LLM in session state")  # Debug
            return False
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        print(f"LLM error details: {str(e)}")  # Debug
        return False

def initialize_components():
    """Initialize all components for RAG functionality"""
    try:
        print("Starting initialize_components")  # Debug

        # Initialize Qdrant
        if not initialize_qdrant():
            st.error("Qdrant initialization failed, stopping execution.")
            print("Failed: Qdrant initialization")  # Debug
            return False

        # Initialize Embeddings
        if not initialize_embeddings():
            st.error("Embeddings initialization failed, stopping execution.")
            print("Failed: Embeddings initialization")  # Debug
            return False

        # Initialize LLM
        if not initialize_llm():
            st.error("LLM initialization failed, stopping execution.")
            print("Failed: LLM initialization")  # Debug
            return False

        # Set global settings
        Settings.embed_model = st.session_state["embeddings"]

        print(f"Settings.llm: {type(Settings.llm)}")  # Debug
        print(f"Settings.embed_model: {type(Settings.embed_model)}")  # Debug
        print(f"Session state after initialization: {list(st.session_state.keys())}")  # Debug

        return True
    except Exception as e:
        st.error(f"Component initialization failed: {str(e)}")
        print(f"Component initialization error: {str(e)}")  # Debug
        return False

def check_existing_vectorstore() -> bool:
    """Check if vectorstore already exists and is accessible"""
    try:
        if "qdrant_client" not in st.session_state:
            return False

        client = st.session_state["qdrant_client"]
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in collection_names:
            return False

        # Check if collection has vectors
        collection_info = client.get_collection(COLLECTION_NAME)
        if collection_info.points_count > 0:
            # Create LangChain vectorstore
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=st.session_state["embeddings"]
            )
            st.session_state["vectorstore"] = vectorstore
            st.session_state["documents_indexed"] = True
            print(f"Existing vectorstore found with {collection_info.points_count} vectors")
            return True

        return False
    except Exception as e:
        print(f"Error checking existing vectorstore: {str(e)}")
        return False

def parse_document(document: any) -> List[str]:
    """Parse document into text chunks"""
    try:
        print(f"Parsing document: {type(document)}")  # Debug
        if isinstance(document, str):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(document)
            print(f"Parsed string into {len(chunks)} chunks")  # Debug
            return chunks
        elif hasattr(document, 'read'):
            if document.name.endswith(".pdf"):
                try:
                    content = document.read()
                    document.seek(0)
                    text = pymupdf4llm.to_markdown(content)
                    cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_text(cleaned_text)
                    print(f"Parsed PDF into {len(chunks)} chunks using pymupdf4llm")  # Debug
                    return chunks
                except Exception as e:
                    print(f"pymupdf4llm failed: {str(e)}, falling back to pymupdf")  # Debug
                    import pymupdf
                    document.seek(0)
                    doc = pymupdf.open(stream=document.read(), filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text("text")
                    cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_text(cleaned_text)
                    print(f"Parsed PDF into {len(chunks)} chunks using pymupdf")  # Debug
                    doc.close()
                    return chunks
            elif document.name.endswith(".txt"):
                text = document.read().decode("utf-8")
                cleaned_text = bleach.clean(text, tags=["p", "b", "i", "strong", "em"], strip=True)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(cleaned_text)
                print(f"Parsed TXT into {len(chunks)} chunks")  # Debug
                return chunks
            else:
                raise ValueError("Unsupported file type. Use .txt or .pdf.")
        else:
            raise Exception(f"Invalid document type: {type(document)}")
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        print(f"Parse document error: {str(e)}")  # Debug
        raise Exception(f"Error parsing document: {str(e)}")

def create_qdrant_vectorstore(text_chunks: List[str]) -> tuple:
    """Create or update Qdrant vectorstore with text chunks"""
    try:
        print("Starting create_qdrant_vectorstore")  # Debug
        if "qdrant_client" not in st.session_state:
            raise ValueError("Qdrant client not initialized")

        client = st.session_state["qdrant_client"]
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"Existing Qdrant collections: {collection_names}")  # Debug

        if COLLECTION_NAME not in collection_names:
            st.info(f"Creating new Qdrant collection '{COLLECTION_NAME}' with dimension 384...")
            print(f"Creating Qdrant collection: {COLLECTION_NAME}")  # Debug
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            time.sleep(5)
            st.success(f"Collection '{COLLECTION_NAME}' created successfully!")

        print(f"Text chunks to add: {len(text_chunks)}")  # Debug
        if text_chunks:
            print(f"Sample chunk: {text_chunks[0][:100]}...")  # Debug

        # Create LangChain vectorstore
        langchain_vectorstore = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=st.session_state["embeddings"]
        )

        # Create LlamaIndex vectorstore
        llamaindex_vectorstore = LlamaQdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME
        )

        # Add text chunks if provided
        if text_chunks:
            print("Adding text chunks to vectorstore")  # Debug
            langchain_vectorstore.add_texts(texts=text_chunks)
            print("Upsert complete, vectorstore updated")  # Debug

        return langchain_vectorstore, llamaindex_vectorstore
    except Exception as e:
        st.error(f"Error creating Qdrant vector store: {str(e)}")
        print(f"Vectorstore creation error: {str(e)}\n{traceback.format_exc()}")  # Debug
        raise Exception(f"Error creating Qdrant vector store: {str(e)}")

def build_sentence_window_index(document_text: str) -> VectorStoreIndex:
    """Build sentence window index for advanced RAG"""
    try:
        print("Starting build_sentence_window_index")  # Debug
        if not st.session_state.get("embeddings"):
            raise Exception("Embeddings not initialized")

        if not document_text.strip():
            raise Exception("Document text is empty")

        print(f"Document text length: {len(document_text)}")  # Debug

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        document = Document(text=document_text)
        print("Parsing nodes")  # Debug
        nodes = node_parser.get_nodes_from_documents([document])
        print(f"Sentence window nodes created: {len(nodes)}")  # Debug

        if not nodes:
            raise Exception("No nodes created for sentence window index")

        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=st.session_state["embeddings"]
        )
        print("Sentence window index created")  # Debug
        return index
    except Exception as e:
        st.error(f"Error building sentence window index: {str(e)}")
        print(f"Sentence window index error: {str(e)}\n{traceback.format_exc()}")  # Debug
        raise Exception(f"Error building sentence window index: {str(e)}")

def build_automerging_index(document_text: str) -> VectorStoreIndex:
    """Build auto-merging index for advanced RAG"""
    try:
        print("Starting build_automerging_index")  # Debug
        if not st.session_state.get("embeddings"):
            raise Exception("Embeddings not initialized")

        if not document_text.strip():
            raise Exception("Document text is empty")

        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        document = Document(text=document_text)

        print("Parsing nodes")  # Debug
        nodes = node_parser.get_nodes_from_documents([document])
        leaf_nodes = get_leaf_nodes(nodes)
        print(f"Auto-merging leaf nodes created: {len(leaf_nodes)}")  # Debug

        if not leaf_nodes:
            raise Exception("No leaf nodes created for auto-merging index")

        llamaindex_vectorstore = st.session_state.get("llamaindex_vectorstore")
        if not llamaindex_vectorstore:
            raise Exception("LlamaIndex vectorstore not initialized")

        storage_context = StorageContext.from_defaults(vector_store=llamaindex_vectorstore)
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=st.session_state["embeddings"]
        )
        print("Auto-merging index created")  # Debug
        return index
    except Exception as e:
        st.error(f"Error building auto-merging index: {str(e)}")
        print(f"Auto-merging index error: {str(e)}\n{traceback.format_exc()}")  # Debug
        raise Exception(f"Error building auto-merging index: {str(e)}")

def index_document(text_or_file: any) -> bool:
    """Index document with full RAG pipeline"""
    try:
        print(f"Starting index_document with input type: {type(text_or_file)}")  # Debug

        if not initialize_components():
            st.error("Failed to initialize RAG components")
            return False

        text_chunks = parse_document(text_or_file)
        print(f"Number of text chunks: {len(text_chunks)}")  # Debug

        if not text_chunks:
            st.error("No text chunks created from document.")
            return False

        document_text = "\n".join(text_chunks)
        print(f"Document text length: {len(document_text)}")  # Debug

        if not document_text.strip():
            st.error("Document text is empty after processing.")
            return False

        print("Creating Qdrant vectorstore")  # Debug
        langchain_vectorstore, llamaindex_vectorstore = create_qdrant_vectorstore(text_chunks)
        st.session_state["vectorstore"] = langchain_vectorstore
        st.session_state["llamaindex_vectorstore"] = llamaindex_vectorstore

        print("Building sentence window index")  # Debug
        sentence_window_index = build_sentence_window_index(document_text)

        print("Building auto-merging index")  # Debug
        automerging_index = build_automerging_index(document_text)

        st.session_state["sentence_window_index"] = sentence_window_index
        st.session_state["automerging_index"] = automerging_index
        st.session_state["document_text"] = document_text
        st.session_state["documents_indexed"] = True

        print("Document indexed successfully")  # Debug
        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        print(f"Indexing error details: {str(e)}\n{traceback.format_exc()}")  # Debug
        return False

def clear_indexes():
    """Clear all indexes and vectorstore"""
    try:
        print("Starting clear_indexes")  # Debug
        client = st.session_state.get("qdrant_client")
        if client:
            collections = client.get_collections()
            if COLLECTION_NAME in [c.name for c in collections.collections]:
                client.delete_collection(COLLECTION_NAME)
                print(f"Qdrant collection {COLLECTION_NAME} deleted")  # Debug

        keys_to_clear = [
            "sentence_window_index",
            "automerging_index",
            "vectorstore",
            "llamaindex_vectorstore",
            "document_text",
            "documents_indexed"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)

        print("Indexes cleared successfully")  # Debug
    except Exception as e:
        st.error(f"Error clearing indexes: {str(e)}")
        print(f"Clear indexes error: {str(e)}")  # Debug

def has_indexed_documents() -> bool:
    """Check if there are indexed documents available"""
    try:
        if st.session_state.get("documents_indexed"):
            return True

        if (st.session_state.get("sentence_window_index") and
                st.session_state.get("automerging_index")):
            return True

        if "qdrant_client" in st.session_state:
            client = st.session_state["qdrant_client"]
            collections = client.get_collections()
            if COLLECTION_NAME in [c.name for c in collections.collections]:
                collection_info = client.get_collection(COLLECTION_NAME)
                if collection_info.points_count > 0:
                    st.session_state["documents_indexed"] = True
                    return True

        return False
    except Exception as e:
        print(f"Error checking indexed documents: {str(e)}")
        return False

def retrieve_and_generate(query: str, context_text: Optional[str] = None) -> str:
    """Retrieve and generate response using RAG pipeline"""
    try:
        print(f"Starting retrieve_and_generate with query: {query[:50]}...")  # Debug

        if not isinstance(Settings.llm, CustomLLM):
            if "llm" in st.session_state:
                Settings.llm = CustomLLM(st.session_state["llm"])
            else:
                return "LLM not properly initialized. Please configure the brain first."

        if not st.session_state.get("sentence_window_index") or not st.session_state.get("automerging_index"):
            print("No advanced indexes found, using basic document chat")  # Debug
            document_text = st.session_state.get("uploaded_document", "")
            if not document_text and not context_text:
                return "No document or context provided. Please upload a document or provide context."

            if document_text:
                response = st.session_state["llm"].document_chat(
                    document_text=document_text,
                    question=query,
                    language=st.session_state.get("chat_language", "English"),
                    glossary=st.session_state.get("glossary")
                )
            else:
                prompt = AppConfig.ENGLISH_CHAT_PROMPT.format(
                    glossary_section="",
                    document_text=context_text or "",
                    question=query
                )
                response = st.session_state["llm"].generate(prompt, temperature=0.3)

            return response

        sentence_index = st.session_state["sentence_window_index"]
        auto_merging_index = st.session_state["automerging_index"]

        print("Creating retrievers")  # Debug
        sw_retriever = sentence_index.as_retriever(similarity_top_k=6)
        am_retriever = AutoMergingRetriever(
            auto_merging_index.as_retriever(similarity_top_k=12),
            storage_context=auto_merging_index.storage_context,
            verbose=True
        )

        print("Creating reranker")  # Debug
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=4
        )

        print("Creating query engines")  # Debug
        sw_query_engine = RetrieverQueryEngine.from_args(
            retriever=sw_retriever,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                reranker
            ],
            llm=Settings.llm
        )
        am_query_engine = RetrieverQueryEngine.from_args(
            retriever=am_retriever,
            node_postprocessors=[reranker],
            llm=Settings.llm
        )

        print("Executing queries")  # Debug
        sw_response = str(sw_query_engine.query(query))
        am_response = str(am_query_engine.query(query))

        final_response = f"{sw_response}\n\nAdditional Insights:\n{am_response}"
        if context_text:
            final_response += f"\n\nContext Provided:\n{context_text}"

        chat_language = st.session_state.get("chat_language", "English")
        if chat_language == "Urdu":
            final_response = st.session_state["llm"].translate_to_urdu(final_response)
        elif chat_language == "Roman Urdu":
            final_response = st.session_state["llm"].translate_to_roman_urdu(final_response)

        return final_response
    except Exception as e:
        st.error(f"Error in retrieve_and_generate: {str(e)}")
        print(f"Retrieve and generate error: {str(e)}")  # Debug
        return f"Error: {str(e)}"

# Auto-initialize on import if not already done
if "rag_initialized" not in st.session_state:
    if initialize_components():
        if check_existing_vectorstore():
            st.session_state["rag_initialized"] = True
            print("RAG system auto-initialized with existing vectorstore")
        else:
            st.session_state["rag_initialized"] = "partial"
            print("RAG components initialized, no existing vectorstore found")
    else:
        st.session_state["rag_initialized"] = False
        print("RAG initialization failed")