import os
import uuid
import json
import weaviate
import time
import fitz  # PyMuPDF
import tempfile # For handling temporary uploaded files
from contextlib import asynccontextmanager

from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel # For request body validation
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration ---
SESSIONS_FILE = "sessions.json" # For storing session IDs and descriptions
CHUNK_SIZE = 500

# --- Global Variables for Clients ---
embedding_model = None
groq_client = None
weaviate_client_instance = None
WEAVIATE_CLUSTER_URL = None
WEAVIATE_API_KEY_VALUE = None
GROQ_API_KEY_VALUE = None


# called within lifespan ---
def initialize_dependencies():
    global embedding_model, groq_client, weaviate_client_instance, WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY_VALUE, GROQ_API_KEY_VALUE

    print("Loading environment variables...")
    load_dotenv() # Loads variables from .env in the current directory

    # ---- START OF ESSENTIAL DEBUG BLOCK ----
    loaded_url = os.getenv('WEAVIATE_URL')
    loaded_api_key = os.getenv('WEAVIATE_API_KEY')
    loaded_groq_key = os.getenv('GROQ_API_KEY')

    print(f"DEBUG PRINT: Value loaded for WEAVIATE_URL from .env: '{loaded_url}'")

    if loaded_api_key:
        print(f"DEBUG PRINT: Value loaded for WEAVIATE_API_KEY from .env: '{loaded_api_key[:5]}...{loaded_api_key[-5:]}'")
    else:
        print("DEBUG PRINT: WEAVIATE_API_KEY was NOT loaded from .env (is None or empty)!")

    if loaded_groq_key:
        print(f"DEBUG PRINT: Value loaded for GROQ_API_KEY from .env: '{loaded_groq_key[:5]}...{loaded_groq_key[-5:]}'")
    else:
        print("DEBUG PRINT: GROQ_API_KEY was NOT loaded from .env (is None or empty)!")
    # ---- END OF ESSENTIAL DEBUG BLOCK ----

    WEAVIATE_CLUSTER_URL = loaded_url
    WEAVIATE_API_KEY_VALUE = loaded_api_key
    GROQ_API_KEY_VALUE = loaded_groq_key

    print("Validating environment variables...")
    # Critical Check:
    if not WEAVIATE_CLUSTER_URL or not WEAVIATE_API_KEY_VALUE or not GROQ_API_KEY_VALUE:
        missing_vars_list = []
        if not WEAVIATE_CLUSTER_URL: missing_vars_list.append("WEAVIATE_URL")
        if not WEAVIATE_API_KEY_VALUE: missing_vars_list.append("WEAVIATE_API_KEY")
        if not GROQ_API_KEY_VALUE: missing_vars_list.append("GROQ_API_KEY")
        error_message = f"CRITICAL ERROR: One or more environment variables are missing after attempting to load from .env: {', '.join(missing_vars_list)}"
        print(error_message)
        raise EnvironmentError(error_message)
    
    print(f"Attempting to connect to Weaviate at: {WEAVIATE_CLUSTER_URL[:30] if WEAVIATE_CLUSTER_URL else 'URL_NOT_LOADED'}...")

    print("Initializing SentenceTransformer model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("SentenceTransformer model loaded.")

    print("Initializing Groq client...")
    groq_client = Groq(api_key=GROQ_API_KEY_VALUE)
    print("Groq client initialized.")

    print("Initializing Weaviate client...")
    try:
        weaviate_client_instance = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_CLUSTER_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY_VALUE),
        )
        print("Weaviate client object created. Checking connection status...")
        if weaviate_client_instance.is_ready():
            print("Successfully connected to Weaviate.")
        else:
            raise RuntimeError("Weaviate client is not ready. Check URL, API key, and network.")
    except Exception as e:
        print(f"Failed to initialize Weaviate client: {e}")
        raise RuntimeError(f"Weaviate client initialization failed: {str(e)}") from e

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

        # Ensure sessions.json can be created/accessed
        try:
            if not os.path.exists(SESSIONS_FILE):
                with open(SESSIONS_FILE, "w") as f:
                    json.dump({}, f)
                print(f"Created empty sessions file: {SESSIONS_FILE}")
            else:
                 with open(SESSIONS_FILE, "r+") as f:
                    content = f.read()
                    if not content.strip():
                        f.seek(0)
                        f.truncate()
                        json.dump({}, f)
                        print(f"Initialized empty or blank sessions file: {SESSIONS_FILE}")
                    else:
                        try:
                            f.seek(0)
                            json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: {SESSIONS_FILE} contains invalid JSON. Re-initializing as empty.")
                            f.seek(0)
                            f.truncate()
                            json.dump({}, f)
        except IOError as e:
             print(f"Warning: Could not ensure {SESSIONS_FILE} exists, is writable, or initialize it: {e}")

        print("FastAPI application starting with session-based PDF uploads.")
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

# --- Session Management (for session-specific data) ---
class SessionManager:
    @staticmethod
    def load_sessions():
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, "r") as f:
                    content = f.read()
                    if not content.strip(): return {}
                    return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading {SESSIONS_FILE}: {e}. Treating as empty.")
                return {}
        return {}

    @staticmethod
    def save_sessions(sessions_data):
        try:
            with open(SESSIONS_FILE, "w") as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            print(f"Error saving {SESSIONS_FILE}: {e}")

    @staticmethod
    def session_exists(session_id: str) -> bool:
        sessions = SessionManager.load_sessions()
        return session_id in sessions

# --- Weaviate Schema ---
COLLECTION_NAME = "Document"

def create_or_update_schema():
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized. Cannot create/update schema.")
        raise RuntimeError("Weaviate client not initialized for schema creation.")

    print(f"Checking if collection '{COLLECTION_NAME}' exists...")
    if weaviate_client_instance.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Verifying properties...")
        try:
            collection_config = weaviate_client_instance.collections.get(COLLECTION_NAME).config.get()
            prop_exists = any(p.name == "session_id" for p in collection_config.properties)
            if not prop_exists:
                print(f"Warning: Collection '{COLLECTION_NAME}' exists but 'session_id' property is missing. Manual schema adjustment might be needed if this is an old schema.")
            else:
                print("'session_id' property found in existing schema.")
        except Exception as e:
            print(f"Could not verify properties of existing collection '{COLLECTION_NAME}': {e}")
        return

    print(f"Creating collection: {COLLECTION_NAME}")
    try:
        weaviate_client_instance.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            properties=[
                weaviate.classes.config.Property(
                    name="session_id",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="ID of the session to which this document chunk belongs",
                ),
                weaviate.classes.config.Property(
                    name="document_uuid",
                    data_type=weaviate.classes.config.DataType.TEXT,
                    description="UUID of the specific document ingestion instance"
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
        full_text_parts = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks", sort=True)
            page_text_segments = []
            for block in blocks:
                if block[6] == 0: # block_type 0 is text
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

# --- API Endpoints ---

class SessionCreationRequest(BaseModel):
    description: str

@app.post("/sessions", summary="Create a new session ID with a description")
async def create_session(request: SessionCreationRequest):
    sessions = SessionManager.load_sessions()
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = request.description
    SessionManager.save_sessions(sessions)
    print(f"Created new session: ID={new_session_id}, Description='{request.description}'")
    return {"session_id": new_session_id, "description": request.description}

@app.get("/sessions", summary="List all session IDs and their descriptions")
async def list_sessions():
    return SessionManager.load_sessions()

@app.get("/sessions/{session_id}", summary="Get description for a specific session ID")
async def get_session_description(session_id: str):
    sessions = SessionManager.load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return {"session_id": session_id, "description": sessions[session_id]}

@app.post("/upload-pdf/{session_id}", summary="Upload a PDF for a specific session ID")
async def upload_pdf_for_session(session_id: str, file: UploadFile = File(...)):
    global weaviate_client_instance, embedding_model
    if not weaviate_client_instance or not embedding_model:
        raise HTTPException(status_code=503, detail="Server components (Weaviate/Embedding model) not ready.")
    if not SessionManager.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found. Please create the session first using the /sessions endpoint.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
            content = await file.read()
            temp_f.write(content)
            temp_file_path = temp_f.name
        
        print(f"Processing uploaded PDF '{file.filename}' for session '{session_id}' from temp path '{temp_file_path}'")
        doc_instance_uuid = str(uuid.uuid4())
        text_content = pdf_to_text(temp_file_path)

        if not text_content:
            raise HTTPException(status_code=400, detail=f"No text could be extracted from '{file.filename}'.")
        chunks = chunk_text(text_content)
        if not chunks:
            raise HTTPException(status_code=400, detail=f"No text chunks could be generated from '{file.filename}'.")

        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        chunks_added_count = 0
        
        print(f"Preparing to batch insert {len(chunks)} chunks for '{file.filename}' (doc_instance_uuid: {doc_instance_uuid}, session_id: {session_id})...")
        with collection.batch.dynamic() as batch_ctx:
            for idx, chunk_text_content in enumerate(chunks):
                if not chunk_text_content.strip():
                    print(f"Skipping empty chunk at index {idx} for '{file.filename}' (session: {session_id}).")
                    continue
                try:
                    vector = embedding_model.encode(chunk_text_content).tolist()
                    batch_ctx.add_object(
                        properties={
                            "session_id": session_id,
                            "document_uuid": doc_instance_uuid,
                            "filename": file.filename,
                            "text": chunk_text_content,
                            "chunk_index": idx
                        },
                        vector=vector
                    )
                    chunks_added_count += 1
                except Exception as e_chunk:
                    print(f"Error processing or batching chunk {idx} of '{file.filename}' for session '{session_id}': {e_chunk}. Skipping this chunk.")
        
        if chunks_added_count > 0:
            print(f"Successfully batched and sent {chunks_added_count} chunks for '{file.filename}' (session '{session_id}') to Weaviate.")
        else:
             print(f"No chunks were successfully added to Weaviate for '{file.filename}' in session '{session_id}'.")
        
        return {
            "message": f"PDF '{file.filename}' processed for session '{session_id}'.",
            "session_id": session_id,
            "filename": file.filename,
            "document_instance_uuid": doc_instance_uuid,
            "chunks_added": chunks_added_count
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error uploading PDF for session '{session_id}', file '{file.filename}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF '{file.filename}': {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e_clean:
                print(f"Error cleaning up temporary file {temp_file_path}: {e_clean}")

@app.get("/query/{session_id}", summary="Query documents within a specific session using a question")
async def query_document_in_session(session_id: str, question: str):
    global weaviate_client_instance, embedding_model, groq_client
    if not weaviate_client_instance or not embedding_model or not groq_client:
        print("Error: One or more clients (Weaviate, Embedding, Groq) not initialized.")
        raise HTTPException(status_code=503, detail="Server components not ready. Please try again shortly.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not SessionManager.session_exists(session_id):
         raise HTTPException(status_code=404, detail=f"Session ID '{session_id}' not found. Please create the session and upload documents first.")

    try:
        print(f"Received query for session '{session_id}': '{question}'")
        print("Embedding query...")
        query_vector = embedding_model.encode(question).tolist()

        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        print(f"Querying Weaviate collection '{COLLECTION_NAME}' for session_id '{session_id}'...")
        
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=3,
            filters=Filter.by_property("session_id").equal(session_id),
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
            return_properties=["text", "filename", "document_uuid", "chunk_index", "session_id"]
        )

        print(f"Retrieved {len(response.objects)} objects from Weaviate for session '{session_id}'.")
        if not response.objects:
            return {"answer": f"I could not find any relevant information in the documents for session '{session_id}' to answer your question.", "retrieved_context_summary": "No relevant chunks found for this session."}

        context_parts = []
        retrieved_filenames = set()
        for obj in response.objects:
            context_parts.append(
                f"From file '{obj.properties['filename']}' (Chunk {obj.properties['chunk_index']}, Session {obj.properties.get('session_id', 'N/A')}):\n{obj.properties['text']}"
            )
            retrieved_filenames.add(obj.properties['filename'])
            if obj.metadata:
                distance_val = obj.metadata.distance
                distance_str = f"{distance_val:.4f}" if distance_val is not None else "N/A"
                print(f"  - Source: {obj.properties['filename']}, Chunk: {obj.properties['chunk_index']}, Session: {obj.properties.get('session_id')}, Distance: {distance_str}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        print("Sending context and question to Groq for completion...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": f"You are an AI assistant. Based SOLELY on the provided context from documents associated with session '{session_id}', answer the user's question. "
                            "If the context does not contain the answer, clearly state that the information is not found in the provided documents for this session. "
                            "Do not use any external knowledge. Be concise. If helpful, you can mention the source filename(s) from which the information was derived.\n\nCONTEXT:\n" + context},
                {"role": "user", "content": question}
            ],
            model="llama-3.3-70b-versatile"  # UPDATED MODEL
        )

        answer = chat_completion.choices[0].message.content
        print(f"Groq answer for session '{session_id}': {answer[:200]}...")
        
        summary_filenames_list = list(retrieved_filenames)
        if not summary_filenames_list:
            summary_display = "N/A"
        else:
            summary_display = ", ".join(summary_filenames_list[:2])
            if len(summary_filenames_list) > 2:
                summary_display += f" and {len(summary_filenames_list)-2} others"
        
        return {"answer": answer, "retrieved_context_summary": f"{len(response.objects)} chunks from files like '{summary_display}' within session '{session_id}'."}

    except weaviate.exceptions.WeaviateQueryException as wqe:
        print(f"Weaviate query error for session '{session_id}': {wqe}")
        raise HTTPException(status_code=503, detail=f"Could not query Weaviate for session '{session_id}': {str(wqe)}")
    except Exception as e:
        print(f"Error during query endpoint call for session '{session_id}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during query for session '{session_id}': {str(e)}")
    
@app.delete("/documents", summary="Delete all document chunks by session_id and filename")
async def delete_documents_by_session_and_filename(session_id: str, filename: str):
    
    global weaviate_client_instance
    if not weaviate_client_instance:
        print("Error: Weaviate client not initialized during delete request.")
        raise HTTPException(status_code=503, detail="Weaviate client not initialized. Cannot perform deletion.")

    if not session_id or not filename:
        raise HTTPException(status_code=400, detail="Both 'session_id' and 'filename' query parameters are required.")

    print(f"Received request to delete documents with session_id='{session_id}' and filename='{filename}'")

    try:
        collection = weaviate_client_instance.collections.get(COLLECTION_NAME)
        
        where_filter = Filter.all_of([
            Filter.by_property("session_id").equal(session_id),
            Filter.by_property("filename").equal(filename)
        ])

        delete_result = collection.data.delete_many(where=where_filter)

        # Collect individual errors if any
        individual_errors = []
        if delete_result.failed > 0 and hasattr(delete_result, 'objects'): # Check if 'objects' attribute exists
            for obj_report in delete_result.objects:
                if obj_report.status == "FAILED" and obj_report.error:
                    individual_errors.append({
                        "id": str(obj_report.uuid), # Assuming uuid attribute exists for the object identifier
                        "error_message": obj_report.error
                    })
        elif delete_result.failed > 0:
             # If .objects isn't there but failures exist, provide a general failure note
            individual_errors.append({"id": "N/A", "error_message": "Generic failure indicated, but detailed error objects not found in expected structure."})


        response_payload = {
            "message": "Deletion process completed.",
            "session_id_filter": session_id,
            "filename_filter": filename,
            "objects_matched": delete_result.matches,
            "successful_deletions": delete_result.successful,
            "failed_deletions": delete_result.failed,
            "individual_errors": individual_errors # Using the collected list
        }

        if delete_result.failed > 0:
            print(f"Warning: Deletion operation had {delete_result.failed} failures for session_id='{session_id}', filename='{filename}'. Individual Errors: {individual_errors}")
        
        if delete_result.matches == 0:
            response_payload["message"] = "No documents found matching the specified criteria. Nothing was deleted."
            print(f"No documents matched for deletion with session_id='{session_id}', filename='{filename}'.")
        elif delete_result.successful > 0 : # Only print success if some were actually deleted
            print(f"Deletion successful for session_id='{session_id}', filename='{filename}'. Matched: {delete_result.matches}, Deleted: {delete_result.successful}")
            
        return response_payload

    except AttributeError as ae: # Catch the specific error if it happens again
        print(f"AttributeError during deletion result processing: {ae}")
        import traceback
        traceback.print_exc()
        # Fallback response if the structure is still not as expected
        return {
            "message": "Deletion attempted, but an error occurred while processing the results.",
            "detail": str(ae),
            "session_id_filter": session_id,
            "filename_filter": filename,
            # You can try to return basic info from delete_result if available
            "objects_matched_raw": getattr(delete_result, 'matches', 'N/A'),
            "successful_deletions_raw": getattr(delete_result, 'successful', 'N/A'),
            "failed_deletions_raw": getattr(delete_result, 'failed', 'N/A')
        }
    except weaviate.exceptions.WeaviateQueryException as wqe:
        print(f"Weaviate query/connection error during deletion: {wqe}")
        raise HTTPException(status_code=503, detail=f"A Weaviate error occurred during deletion: {str(wqe)}")
    except Exception as e:
        print(f"An unexpected error occurred during deletion for session_id='{session_id}', filename='{filename}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during the deletion process: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)