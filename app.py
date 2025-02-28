import faiss
import numpy as np
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer

# Load a pre-trained transformer model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample news articles dataset
news_articles = [
    "Bitcoin price surges as market rebounds",
    "New AI model outperforms humans in text recognition",
    "Stock market crashes amid economic uncertainty",
    "Scientists discover new planet in habitable zone",
    "Football team wins championship in dramatic final"
]

# Generate embeddings for the articles
embeddings = model.encode(news_articles, convert_to_numpy=True)

# Build FAISS index for similarity search
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)

def recommend_articles(query, top_k=3):
    """Recommend similar articles based on the given query."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [news_articles[i] for i in indices[0]]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            recommendations = recommend_articles(query)
    return render_template('index.html', recommendations=recommendations)

# Ensure static files are correctly served
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)