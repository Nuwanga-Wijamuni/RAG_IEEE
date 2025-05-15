# main.py
import os
import uuid
import json
import weaviate
import time
import fitz  
from contextlib import asynccontextmanager 

from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
# from weaviate.exceptions import WeaviateQueryException 
from fastapi import FastAPI, HTTPException
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration ---
DATA_DIR = "data"
METADATA_FILE = "metadata.json"
CHUNK_SIZE = 500  

# --- Global Variables for Clients ---
embedding_model = None
groq_client = None
weaviate_client_instance = None 
WEAVIATE_CLUSTER_URL = None 
WEAVIATE_API_KEY_VALUE = None 
GROQ_API_KEY_VALUE = None 


#  called within lifespan ---
def initialize_dependencies():
 
    global embedding_model, groq_client, weaviate_client_instance, WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY_VALUE, GROQ_API_KEY_VALUE

    print("Loading environment variables...")
    load_dotenv()

    print("Validating environment variables...")
    required_env_vars = ["WEAVIATE_URL", "WEAVIATE_API_KEY", "GROQ_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

    WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY_VALUE = os.getenv("WEAVIATE_API_KEY")
    GROQ_API_KEY_VALUE = os.getenv("GROQ_API_KEY")

    print(f"Attempting to connect to Weaviate at: {WEAVIATE_CLUSTER_URL[:30]}...")

    print("Initializing SentenceTransformer model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("SentenceTransformer model loaded.")

    print("Initializing Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY_VALUE)
    print("Groq client initialized.")

    print("Initializing Weaviate client...")
    weaviate_client_instance = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_CLUSTER_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY_VALUE),
    )
    print("Weaviate client object created. Checking connection status...")
    if weaviate_client_instance.is_ready():
        print("Successfully connected to Weaviate.")
    else:
        
        raise RuntimeError("Weaviate client is not ready. Check URL, API key, and network.")

def close_weaviate_connection():
    """Closes the Weaviate client connection."""
    global weaviate_client_instance
    if weaviate_client_instance:
        print("Closing Weaviate client connection...")
        try:
            weaviate_client_instance.close()
            print("Weaviate client connection closed.")
        except Exception as e:
            print(f"Error closing Weaviate connection: {e}")


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application startup: Initializing dependencies...")
    try:
        initialize_dependencies()
        create_or_update_schema() 

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            print(f"Data directory '{DATA_DIR}' created during startup.")
        print(f"FastAPI application starting. Ensure '{DATA_DIR}' contains your PDF files.")
        print(f"If you have changed Weaviate clusters or want a fresh start, ensure '{METADATA_FILE}' was deleted before this run to re-ingest all data.")
        print("Application startup complete.")
        yield 
    except Exception as e:
        print(f"An error occurred during application startup: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        raise RuntimeError(f"Startup initialization failed: {str(e)}") from e

    finally:
        # Shutdown
        print("Application shutdown: Cleaning up resources...")
        close_weaviate_connection()
        print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan) 

# --- Metadata Management ---
class MetadataManager:
    @staticmethod
    def load():
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r") as f:
                    content = f.read()
                    if not content.strip(): # File is empty
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: {METADATA_FILE} contains invalid JSON. Treating as empty.")
                return {}
            except Exception as e:
                print(f"Error loading {METADATA_FILE}: {e}. Treating as empty.")
                return {}
        return {}

    @staticmethod
    def save(metadata):
        try:
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving {METADATA_FILE}: {e}")


# --- Weaviate Schema ---
COLLECTION_NAME = "Document" 

def create_or_update_schema():
    
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized. Cannot create/update schema.")
        
        raise RuntimeError("Weaviate client not initialized for schema creation.")

    print(f"Checking if collection '{COLLECTION_NAME}' exists...")
    if weaviate_client_instance.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    print(f"Creating collection: {COLLECTION_NAME}")
    try:
        weaviate_client_instance.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(), # Using external embedding model
            properties=[
                weaviate.classes.config.Property(
                    name="document_uuid", # UUID of the source document file
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="UUID of the source document"
                ),
                weaviate.classes.config.Property(
                    name="filename",
                    data_type=weaviate.classes.config.DataType.TEXT,
                ),
                weaviate.classes.config.Property(
                    name="text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                ),
                weaviate.classes.config.Property(
                    name="chunk_index",
                    data_type=weaviate.classes.config.DataType.INT,
                )
            ]
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Error creating collection '{COLLECTION_NAME}': {e}")
        raise 

# --- PDF Processing ---
def pdf_to_text(file_path: str) -> str:
  
    print(f"Processing PDF with PyMuPDF: {file_path}")
    try:
        doc = fitz.open(file_path)  
        full_text_parts = [] # List to hold text from each page

        for page_num, page in enumerate(doc):
            
            blocks = page.get_text("blocks", sort=True)
            page_text_segments = [] 

            for block in blocks: 
                if block[6] == 0:  
                    page_text_segments.append(block[4]) # block[4] is the text
            page_text = "\n".join(page_text_segments)

            if page_text.strip(): 
                full_text_parts.append(page_text)
            else:
                print(f"Warning: No text extracted from page {page_num+1} of {os.path.basename(file_path)} using PyMuPDF 'blocks' method.")
        
        doc.close() 
        full_text = "\n\n".join(full_text_parts) 
        
        if not full_text.strip():
            print(f"Warning: No text content found in {os.path.basename(file_path)} after processing all pages with PyMuPDF.")
        return full_text
    except Exception as e:
        print(f"Error reading PDF {os.path.basename(file_path)} with PyMuPDF: {e}")
        if "cannot open" in str(e).lower() or "format error" in str(e).lower() or "no objects found" in str(e).lower():
             print(f"PyMuPDF could not open or process the file. It might be corrupted, not a valid PDF, or empty: {file_path}")
        return ""

def chunk_text(text: str) -> list[str]:
   
    if not text.strip():
        return []
    words = text.split() 
    chunks = [' '.join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    print(f"Text chunked into {len(chunks)} chunks of approximately {CHUNK_SIZE} words each.")
    return chunks

# --- Weaviate Operations ---
def add_document(file_path: str):
    
    global weaviate_client_instance, embedding_model
    if not weaviate_client_instance or not embedding_model:
        print("Error: Weaviate client or embedding model not initialized. Cannot add document.")
        return 0

    filename = os.path.basename(file_path)
    metadata = MetadataManager.load()

    if filename in metadata:
        print(f"Document '{filename}' found in metadata (UUID: {metadata[filename]}). Skipping processing.")
        return 0

    print(f"Adding new document: {filename}")
    doc_uuid = str(uuid.uuid4()) 
    text_content = pdf_to_text(file_path) 

    if not text_content: 
        print(f"No text extracted from '{filename}'. Skipping.")
        return 0

    chunks = chunk_text(text_content)
    if not chunks: 
        print(f"No chunks generated from '{filename}'. Skipping.")
        return 0

    collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
    chunks_added_count = 0

    print(f"Preparing to batch insert {len(chunks)} chunks for '{filename}' (UUID: {doc_uuid})...")
    
    try:
        with collection.batch.dynamic() as batch_ctx:
            for idx, chunk_text_content in enumerate(chunks):
                if not chunk_text_content.strip(): 
                    print(f"Skipping empty chunk at index {idx} for '{filename}'.")
                    continue
                try:
                    vector = embedding_model.encode(chunk_text_content).tolist()
                    batch_ctx.add_object(
                        properties={
                            "document_uuid": doc_uuid,
                            "filename": filename,
                            "text": chunk_text_content,
                            "chunk_index": idx
                        },
                        vector=vector
                    )
                    chunks_added_count += 1
                except Exception as e:
                    print(f"Error processing or batching chunk {idx} of '{filename}': {e}. Skipping this chunk.")
    except Exception as e:
        print(f"An error occurred during the batch operation for '{filename}': {e}")
        return 0 

    if chunks_added_count > 0:
        print(f"Successfully batched and sent {chunks_added_count} chunks for '{filename}' to Weaviate.")
        metadata[filename] = doc_uuid
        MetadataManager.save(metadata)
    else:
        print(f"No chunks were successfully batched for '{filename}'. Metadata not updated.")

    return chunks_added_count

def delete_document(filename: str):
    
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized. Cannot delete document.")
        return False

    print(f"Attempting to delete document: {filename}")
    metadata = MetadataManager.load()
    if filename not in metadata:
        print(f"Document '{filename}' not found in metadata. Cannot delete from Weaviate.")
        return False

    doc_uuid = metadata[filename]
    print(f"Deleting document '{filename}' (UUID: {doc_uuid}) from Weaviate.")
    collection = weaviate_client_instance.collections.get(COLLECTION_NAME)

    try:
        delete_result = collection.data.delete_many(
            where=Filter.by_property("document_uuid").equal(doc_uuid)
        )
        print(f"Weaviate delete_many result for '{filename}':")
        print(f"  Successful: {delete_result.successful}")
        print(f"  Failed: {delete_result.failed}")
        print(f"  Matches: {delete_result.matches}") 
        
        if delete_result.successful > 0 or (delete_result.matches > 0 and delete_result.failed == 0):
            print(f"Successfully deleted data for '{filename}' (UUID: {doc_uuid}) from Weaviate.")
            if filename in metadata: 
                 del metadata[filename]
                 MetadataManager.save(metadata)
            return True
        elif delete_result.matches == 0:
            print(f"No objects found in Weaviate for document_uuid '{doc_uuid}' ('{filename}'). Removing from metadata as it's effectively gone.")
            if filename in metadata: 
                del metadata[filename]
                MetadataManager.save(metadata)
            return True 
        else:
            error_messages = getattr(delete_result, 'errors', []) # Access errors if available
            print(f"Failed to delete all objects for '{filename}' (UUID: {doc_uuid}) from Weaviate. Errors: {error_messages}")
            return False
    except Exception as e:
        print(f"Error during Weaviate delete_many for '{filename}': {e}")
        return False

# --- Synchronization Logic ---
def sync_data_directory():
    
    print("\n--- Starting Data Directory Synchronization ---")
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' does not exist. Creating it.")
        os.makedirs(DATA_DIR)
        print(f"Created data directory '{DATA_DIR}'. Please add your PDF files there.")
        return {"new_files_processed": 0, "chunks_added": 0, "files_deleted_from_weaviate": 0,
                "total_pdfs_in_directory": 0, "total_docs_tracked_in_metadata": 0,
                "message": f"Data directory '{DATA_DIR}' was missing and has been created."}

    metadata = MetadataManager.load()
    print(f"Loaded {len(metadata)} entries from '{METADATA_FILE}'.")

    try:
        current_pdf_files = {f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')}
    except Exception as e:
        print(f"Error listing files in '{DATA_DIR}': {e}")
        
        raise HTTPException(status_code=500, detail=f"Could not read data directory {DATA_DIR}: {e}")


    print(f"Found {len(current_pdf_files)} PDF files in '{DATA_DIR}'.")
    tracked_files_in_metadata = set(metadata.keys())

    files_to_delete = tracked_files_in_metadata - current_pdf_files
    deleted_count = 0
    if files_to_delete:
        print(f"\n--- Processing Deletions ({len(files_to_delete)} files) ---")
        for filename in files_to_delete:
            print(f"File '{filename}' is in metadata but not in '{DATA_DIR}'. Attempting to delete from Weaviate.")
            if delete_document(filename): 
                deleted_count += 1
                print(f"Successfully processed deletion for '{filename}'.")
            else:
                print(f"Failed to properly delete '{filename}'. It might remain in metadata if deletion from Weaviate failed.")
    else:
        print("\nNo files to delete from Weaviate (or metadata).")
    
    
    current_metadata_after_deletions = MetadataManager.load()
    tracked_files_in_metadata_after_deletions = set(current_metadata_after_deletions.keys())
    files_to_add = current_pdf_files - tracked_files_in_metadata_after_deletions

    added_chunks_total = 0
    added_files_count = 0
    if files_to_add:
        print(f"\n--- Processing Additions ({len(files_to_add)} files) ---")
        for filename in files_to_add:
            file_path = os.path.join(DATA_DIR, filename)
            print(f"File '{filename}' is in '{DATA_DIR}' but not in metadata. Attempting to add to Weaviate.")
            num_chunks_added = add_document(file_path) 
            if num_chunks_added > 0:
                added_chunks_total += num_chunks_added
                added_files_count +=1
                print(f"Successfully processed and added '{filename}'.")
            else:
                print(f"No chunks added for '{filename}'. It might be empty, unreadable, or an error occurred during processing.")
    else:
        print("\nNo new files to add to Weaviate.")

    final_metadata = MetadataManager.load() 
    print(f"\n--- Synchronization Complete ---")
    print(f"New files processed and added to Weaviate: {added_files_count} (total {added_chunks_total} chunks).")
    print(f"Files/documents deleted from Weaviate: {deleted_count}.")
    print(f"Total PDF files currently in '{DATA_DIR}': {len(current_pdf_files)}.")
    print(f"Total documents now tracked in '{METADATA_FILE}': {len(final_metadata)}.")

    return {
        "new_files_processed": added_files_count,
        "chunks_added": added_chunks_total,
        "files_deleted_from_weaviate": deleted_count,
        "total_pdfs_in_directory": len(current_pdf_files),
        "total_docs_tracked_in_metadata": len(final_metadata)
    }

# --- API Endpoints ---
@app.post("/sync", summary="Synchronize PDF files from the data directory with Weaviate")
async def trigger_sync():
    
    try:
        sync_result = sync_data_directory()
        return sync_result
    except HTTPException as http_exc: 
        raise http_exc
    except Exception as e:
        print(f"Error during sync endpoint call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during synchronization: {str(e)}")

@app.get("/query", summary="Query documents using a question")
async def query_document(question: str):
    """Endpoint to query ingested documents based on a natural language question."""
    global weaviate_client_instance, embedding_model, groq_client
    if not weaviate_client_instance or not embedding_model or not groq_client:
        print("Error: One or more clients (Weaviate, Embedding, Groq) not initialized.")
        raise HTTPException(status_code=503, detail="Server components not ready. Please try again shortly.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        print(f"Received query: '{question}'")
        print("Embedding query...")
        query_vector = embedding_model.encode(question).tolist()

        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        print(f"Querying Weaviate collection '{COLLECTION_NAME}'...")
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=3, 
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True), 
            return_properties=["text", "filename", "document_uuid", "chunk_index"] 
        )

        print(f"Retrieved {len(response.objects)} objects from Weaviate.")
        if not response.objects:
            return {"answer": "I could not find any relevant information in the documents to answer your question.", "retrieved_context_summary": "No relevant chunks found."}

        context_parts = []
        retrieved_filenames = set()
        for obj in response.objects:
            context_parts.append(
                f"From file '{obj.properties['filename']}' (Chunk {obj.properties['chunk_index']}):\n{obj.properties['text']}"
            )
            retrieved_filenames.add(obj.properties['filename'])
            if obj.metadata: 
                # Corrected distance formatting
                distance_val = obj.metadata.distance
                distance_str = f"{distance_val:.4f}" if distance_val is not None else "N/A"
                print(f"  - Source: {obj.properties['filename']}, Chunk: {obj.properties['chunk_index']}, Distance: {distance_str}")


        context = "\n\n---\n\n".join(context_parts)
        
        print("Sending context and question to Groq for completion...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are an AI assistant. Based SOLELY on the provided context from documents, answer the user's question. "
                            "If the context does not contain the answer, clearly state that the information is not found in the provided documents. "
                            "Do not use any external knowledge. Be concise. If helpful, you can mention the source filename(s) from which the information was derived.\n\nCONTEXT:\n" + context},
                {"role": "user", "content": question}
            ],
            model="llama-3.3-70b-versatile" 
        )

        answer = chat_completion.choices[0].message.content
        print(f"Groq answer: {answer}")
        
        summary_filenames_list = list(retrieved_filenames)
        if not summary_filenames_list: # Handle case where no files were retrieved (though covered by earlier check)
             summary_display = "N/A"
        else:
            summary_display = ", ".join(summary_filenames_list[:2]) 
            if len(summary_filenames_list) > 2:
                summary_display += " and others"
            elif not summary_filenames_list: # Should not happen if response.objects is not empty
                summary_display = "unknown files"


        return {"answer": answer, "retrieved_context_summary": f"{len(response.objects)} chunks from files like '{summary_display}'."}

    except weaviate.exceptions.WeaviateQueryException as wqe: 
        print(f"Weaviate query error: {wqe}")
        raise HTTPException(status_code=503, detail=f"Could not query Weaviate: {str(wqe)}")
    except Exception as e:
        print(f"Error during query endpoint call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 

