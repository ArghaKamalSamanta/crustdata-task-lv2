import faiss
import json
import os
import glob
from create_vectorDB import model
from huggingface_hub import InferenceClient

DBs = []

# Load the FAISS Vector Database
index = faiss.read_index("notion_vector_index.faiss")

# Load metadata (retrieved chunks)
with open("notion_metadata.json", "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)

DBs.append((index, metadata))

for item in os.listdir(os.getcwd()):
    item_path = os.path.join(os.getcwd(), item)
    if os.path.isdir(item_path) and item.startswith("temp_"):
        faiss_file = glob.glob(os.path.join(item_path, "*.faiss"))[0]
        json_file = glob.glob(os.path.join(item_path, "*.json"))[0]

        ind = faiss.read_index(faiss_file)
        with open(json_file, "r", encoding="utf-8") as mt_file:
            mt_data = json.load(mt_file)
        DBs.append((ind, mt_data))

# Huggingface api
client = InferenceClient(api_key="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")

# Function to Retrieve Relevant Chunks
def retrieve_chunks(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    ind_mt_dict = {}
    for ind, mt in DBs:
        distances, indices = ind.search(query_embedding, top_k)
        for i, idx in enumerate(indices[0]):
            ind_mt_dict[distances[0][i]] = mt["chunks"][idx]
            
    results = [ind_mt_dict[key] for key in sorted(ind_mt_dict.keys())[:5]]
    return results

# Generate Output 
def generate_response(query):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(query)
    context = " ".join(retrieved_chunks)

    # Prepare the input 
    input_text = f"""
        CONTEXT: {context}
        
        QUESTION: {query}

        THINGS TO BE NOTED:
        1. ADD PROPER CODES OR COMMANDS IF NECESSARY.
        2. IF NOTHING IS SPECIFIED, AND YOU FIND BOTH THE PYTHON CODE AND CURL COMMANDS ARE EQUALLY APPLICABLE, PREFER CURL COMMANDS OVER THE PYTHON CODE.
        3. IF IMPORTANT LINKS ARE AVAILABLE TO LOOK INTO, ATTACH THEM.
        
        Answer:
        """
    
    messages = [
	    {
	    	"role": "user",
	    	"content": input_text
	    }
    ]

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
    	messages=messages, 
    	max_tokens=1000
    )

    return completion.choices[0].message.content

# Example Query
# if __name__ == "__main__":
#     user_query = "How do I authenticate using the API?"
#     response = generate_response(user_query)
#     print("Generated Response:")
#     print(response)
