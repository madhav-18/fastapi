from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

# Load the LLM for text generation
generator = pipeline('text-generation', model='gpt2')

# Load the sentence transformer for encoding
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample articles (in a real scenario, you'd have a larger corpus)
articles = [
    "Depression is a common mental health disorder characterized by persistent sadness and loss of interest.",
    "Anxiety disorders involve excessive worry and fear about everyday situations.",
    "Cognitive-behavioral therapy (CBT) is an effective treatment for many mental health conditions.",
    "Mindfulness meditation can help reduce stress and improve overall mental well-being.",
]

# Encode articles
article_embeddings = encoder.encode(articles)

# Create FAISS index
index = faiss.IndexFlatL2(article_embeddings.shape[1])
index.add(article_embeddings.astype('float32'))

X = ["I feel sad all the time", "I'm worried about everything", "I can't sleep at night", "I feel great today"]
y = ["depression", "anxiety", "insomnia", "positive"]

# Create and train the classification model
clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
clf.fit(X, y)

app = FastAPI()


class RAGInput(BaseModel):
    prompt: str


class ClassificationInput(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Chatbot API"}


@app.lifespan("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())


@app.post("/rag")
def rag_endpoint(input_data: RAGInput):
    # Encode the input prompt
    query_embedding = encoder.encode([input_data.prompt])

    # Retrieve relevant articles
    k = 2  # Number of articles to retrieve
    distances, indices = index.search(query_embedding.astype('float32'), k)

    relevant_articles = [articles[i] for i in indices[0]]

    # Generate response using the LLM
    context = " ".join(relevant_articles)
    prompt = f"Context: {context}\nUser: {input_data.prompt}\nChatbot:"
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

    return {"response": response, "relevant_articles": relevant_articles}


@app.post("/classification")
def classification_endpoint(input_data: ClassificationInput):
    prediction = clf.predict([input_data.text])[0]
    return {"classification": prediction}
