from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the LLM for text generation
generator = pipeline('text-generation', model='gpt2')

# Load the sentence transformer for encoding
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Load articles from file
with open('data/articles.txt', 'r') as f:
    articles = f.read().splitlines()

# Encode articles
article_embeddings = encoder.encode(articles)

# Create FAISS index
index = faiss.IndexFlatL2(article_embeddings.shape[1])
index.add(article_embeddings.astype('float32'))

def rag_response(prompt):
    # Encode the input prompt
    query_embedding = encoder.encode([prompt])
    
    # Retrieve relevant articles
    k = 2  # Number of articles to retrieve
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    relevant_articles = [articles[i] for i in indices[0]]
    
    # Generate response using the LLM
    context = " ".join(relevant_articles)
    full_prompt = f"Context: {context}\nUser: {prompt}\nChatbot:"
    response = generator(full_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return {"response": response, "relevant_articles": relevant_articles}