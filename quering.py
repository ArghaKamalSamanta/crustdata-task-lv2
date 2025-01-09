import faiss
import json
from create_vectorDB import model

# Load the FAISS index
index = faiss.read_index("notion_vector_index.faiss")

# Query text
query = "How do I authenticate using the API?"
query_embedding = model.encode([query], convert_to_numpy=True)

# Search the database
k = 5  # Number of results to return
distances, indices = index.search(query_embedding, k)

# Load metadata
with open("notion_metadata.json", "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)

# Print results
print("Top results:")
for i, idx in enumerate(indices[0]):
    print(f"\nResult {i + 1} (Distance: {distances[0][i]:.4f}):")
    print(metadata["chunks"][idx])
    
