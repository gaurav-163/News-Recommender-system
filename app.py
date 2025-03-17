import faiss
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
from Articles import articles

# Load a pre-trained transformer model for text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
# Sample news articles dataset
# news_articles = [
#     "Bitcoin price surges as market rebounds",
#     "New AI model outperforms humans in text recognition",
#     "Stock market crashes amid economic uncertainty",
#     "Scientists discover new planet in habitable zone",
#     "Football team wins championship in dramatic final",
#     "Global warming effects worsen, scientists issue urgent warning",
#     "Breakthrough in cancer research offers new hope for patients",
#     "Self-driving cars are now being tested in major cities",
#     "Quantum computing makes significant progress towards real-world applications",
#     "NASA plans new mission to explore the outer solar system",
#     "Cryptocurrency regulations tighten amid security concerns",
#     "Electric vehicles surpass gasoline cars in annual sales",
#     "Advancements in renewable energy technology promise a sustainable future",
#     "Artificial intelligence used to detect early signs of Alzheimer’s disease",
#     "Major cyberattack disrupts global supply chains",
#     "New advancements in gene editing spark ethical debates",
#     "Metaverse adoption grows as virtual workplaces expand",
#     "World Health Organization issues guidelines for post-pandemic recovery",
#     "Mars rover sends back stunning high-resolution images",
#     "Fusion energy breakthroughs bring hope for clean unlimited power",
#     "Smart cities development accelerates with 5G and IoT integration",
#     "Renewable hydrogen production reaches new efficiency records",
#     "Advances in drone technology revolutionize delivery services",
#     "Deepfake technology raises concerns over misinformation and privacy",
#     "New study reveals impact of social media on mental health",
#     "Breakthrough in battery technology extends electric vehicle range",
#     "Ocean cleanup projects show promising results in reducing plastic waste",
#     "Scientists develop AI system to predict earthquakes with high accuracy",
#     "New satellite technology improves global climate monitoring",
#     "Biodegradable packaging solutions gain traction in the industry"
# ]

news_articles = articles
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
