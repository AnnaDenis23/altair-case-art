# altair-case-art
# AI Art Historian: Intelligent Assistant for Preliminary Artwork Attribution

## Project Overview

This project implements an intelligent system for preliminary analysis and attribution of artworks using computer vision and deep learning techniques. The system assists art historians and museum specialists by analyzing uploaded images, generating hypotheses about artwork characteristics (genre, epoch, style), finding visually similar works in reference collections, and providing confidence scores for automated predictions.

The system is designed as a decision-support tool that generates hypotheses for expert review rather than providing definitive attributions.

## Problem Statement

When working with digital museum collections, specialists need to:
- Compare artworks with known works in collections
- Determine possible epoch, artistic direction, and genre
- Identify visually similar pieces for comparative analysis

This system addresses these needs by leveraging state-of-the-art computer vision models to automate preliminary analysis and support expert decision-making.

## Research Component

The project implements and compares two distinct approaches to artwork analysis:

1. **Similarity Search Approach**: Finds visually similar works using embedding-based retrieval and aggregates their metadata to form attribution hypotheses
2. **Direct Classification Approach**: Uses zero-shot classification to directly predict artwork characteristics

Both approaches include confidence estimation and automatic flagging of cases requiring expert review.

## Key Features

- **Automated Attribute Prediction**: Determines genre and epoch of uploaded artworks
- **Visual Similarity Search**: Retrieves top-5 most visually similar works from the reference collection
- **Confidence Scoring**: Provides uncertainty estimates for all predictions
- **Expert Review Flagging**: Automatically identifies low-confidence cases requiring human expert validation
- **Comparative Analysis**: Side-by-side comparison of two analytical approaches
- **Interactive Interface**: Web-based interface for easy image upload and result visualization

## Technology Stack

- **Programming Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch
- **Computer Vision Models**: OpenAI CLIP (ViT-B/32)
- **Machine Learning**: scikit-learn (cosine similarity, metrics)
- **Data Processing**: Pandas, NumPy, Pillow
- **Web Interface**: Streamlit
- **Embedding Storage**: NumPy arrays for efficient similarity search

## Project Structure
ltair-case-art/
├── app.py # Main Streamlit application
├── config.py # Configuration (paths, thresholds, model settings)
├── requirements.txt # Python dependencies
├── components/
│ └── attribution_card.py # UI component for displaying results
├── models/
│ ├── clip_search.py # Similarity search implementation
│ ── classifier.py # Zero-shot classification implementation
── utils/
│ ├── preprocessing.py # Image preprocessing and CLIP integration
│ ── confidence.py # Confidence scoring and expert flagging logic
├── data/ # Reference artwork images
└── files/ # Precomputed embeddings and metadata
├── paintings_embeddings.npy
└── russian_paintings_with_embeddings.csv



## Dataset

The system uses a curated dataset of Russian classical paintings from open museum collections, including:
- Artwork images in JPG format
- Metadata: title, author, year, genre, epoch
- Precomputed visual embeddings for efficient similarity search

The dataset was preprocessed to extract temporal information from titles, classify genres from GPT-generated descriptions, and filter low-quality metadata entries.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd altair-case-art


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


pip install -r requirements.txt

streamlit run app.py