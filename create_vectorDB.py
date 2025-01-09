import os
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Directory containing the .txt files
data_dir = "./notion_pages"  
files = ["detailed_api.txt", "enrichment_api.txt"]

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

db_path = "notion_vector_index.faiss"

if not os.path.exists(db_path):

    # Prepare the data for embedding
    all_chunks = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        text = load_text(file_path)
        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)
    
    print("Chunking done.")
    
    # Generate embeddings for all chunks
    embeddings = generate_embeddings(all_chunks)
    
    # Create and save FAISS vector database
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)  # L2 (Euclidean) similarity index
    index.add(embeddings)  # Add embeddings to the index
    
    # Save the index to a file
    faiss.write_index(index, "notion_vector_index.faiss")
    print(f"Vector database created and saved as 'notion_vector_index.faiss'.")
    
    # Save metadata (chunks)
    import json
    with open("notion_metadata.json", "w", encoding="utf-8") as meta_file:
        json.dump({"chunks": all_chunks}, meta_file, indent=4, ensure_ascii=False)
    
    print("Metadata saved to 'notion_metadata.json'.")
