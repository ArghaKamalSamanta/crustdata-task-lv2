import os
import faiss
import tempfile
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Read text files
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text.strip()

# Split text into manageable chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Generate embeddings
def generate_embeddings(text_chunks):
    return model.encode(text_chunks, convert_to_numpy=True)

def create_vectorDB(files):
    temp_dir = tempfile.mkdtemp(dir=os.getcwd(), prefix="temp_")  # Create a temporary directory
    temp_faiss_path = os.path.join(temp_dir, "temp_vector_index.faiss")
    temp_metadata_path = os.path.join(temp_dir, "temp_metadata.json")

    # Prepare the data for embedding
    all_chunks = []
    for file in files:
        text = file.getvalue().decode("utf-8")
        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)
    
    print("Chunking additional knowledge done.")
    
    # Generate embeddings for all chunks
    embeddings = generate_embeddings(all_chunks)
    
    # Create and save FAISS vector database
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 (Euclidean) similarity index
    index.add(embeddings)  # Add embeddings to the index
    
    # Save the index to a file
    faiss.write_index(index, temp_faiss_path)
    print(f"Additional vector database created and saved as {temp_faiss_path}.")
    
    # Save metadata (chunks)
    import json
    with open(temp_metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump({"chunks": all_chunks}, meta_file, indent=4, ensure_ascii=False)
    
    print(f"Additional metadata saved to {temp_metadata_path}.")
