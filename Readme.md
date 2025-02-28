# News Recommender System

## Introduction
The News Recommender System is a machine learning-based application that suggests relevant news articles based on user queries. It leverages a pre-trained transformer model for text embeddings and FAISS for efficient similarity search.

## Features
- Uses **Sentence Transformers** to generate news article embeddings.
- Implements **FAISS (Facebook AI Similarity Search)** for fast retrieval of similar news.
- Provides a **Gradio-based** user interface for easy interaction.
- Supports diverse and expanding news topics.

## Installation
To set up and run the News Recommender System, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/news-recommender.git
   cd news-recommender
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the application using:
```bash
python app.py
```
This will launch the Gradio UI, where users can enter a news topic to receive recommendations.

## Code Overview
- **app.py**: Main script containing the machine learning pipeline and Gradio interface.
- **requirements.txt**: Lists required Python dependencies.
- **README.md**: Documentation for understanding and running the project.

## How It Works
1. **Preprocess News Articles**: The system uses a transformer model to convert news articles into embeddings.
2. **Build FAISS Index**: FAISS is used for fast similarity searching.
3. **Query Processing**: User input is encoded and searched in the FAISS index.
4. **Recommendations Displayed**: The top similar articles are returned as recommendations.

## Dependencies
Ensure you have the following installed:
- `faiss`
- `numpy`
- `gradio`
- `sentence-transformers`

Install them using:
```bash
pip install faiss numpy gradio sentence-transformers
```

## Future Enhancements
- Integrate live news feeds for real-time recommendations.
- Improve the model with domain-specific training.
- Deploy as a web application using Flask/FastAPI.

## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss.

