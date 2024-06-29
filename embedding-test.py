import ollama
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Assuming 'collection' is the name of your ChromaDB collection with documents and embeddings
collection_name = "article_embeddings"
collection = chroma_client.get_collection(collection_name)

# Step 2: Retrieve the most relevant document
prompt = "What animals are llamas related to?"

# Generate an embedding for the prompt
response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
)
# Query the collection for the most relevant document
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
# Retrieve the most relevant document
data = results['documents'][0][0]

# Step 3: Generate an answer using the prompt and retrieved document
output = ollama.generate(
    model="llama2",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

# Print the generated response
print(output['response'])