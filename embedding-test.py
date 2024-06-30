import ollama
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./output/chromadb")
collection = chroma_client.get_or_create_collection(name="article_embeddings")

# Step 2: Retrieve the most relevant document
prompt = "Tell me about the Ogallala Aquifer."

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
promptTemplate = f"Using this data: {data}. Respond to this prompt: {prompt}"
output = ollama.generate(
    model="llama3",
    prompt=promptTemplate
)

# Print the prompt
print("Prompt:{promptTemplate}")

# Print the generated response
print(output['response'])