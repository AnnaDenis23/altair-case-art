import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import config
from utils.preprocessing import resolve_image_path

# Глобальные переменные для хранения данных (загружаются один раз)
_embeddings = None
_metadata = None

def load_data():
    global _embeddings, _metadata
    if _embeddings is None:
        _embeddings = np.load(config.EMBEDDINGS_PATH)
        _metadata = pd.read_csv(config.METADATA_PATH)
        # Исправляем пути к картинкам в метаданных
        _metadata['local_path'] = _metadata['filename'].apply(resolve_image_path)

def find_similar(query_embedding, top_k=config.TOP_K_SIMILAR):
    """Ищет top_k самых похожих картин по косинусному сходству."""
    load_data()
    
    # Переводим тензор в numpy для sklearn
    query_np = query_embedding.cpu().numpy().reshape(1, -1)
    
    # Считаем сходство
    similarities = cosine_similarity(query_np, _embeddings)[0]
    
    # Получаем индексы топ-K (отсортированные по убыванию)
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    similar_items = []
    for idx in top_indices:
        row = _metadata.iloc[idx]
        similar_items.append({
            "title": row.get('ru_title', 'Без названия'),
            "author": row.get('ru_author', 'Неизвестен'),
            "genre": row.get('genre', 'Неизвестно'),
            "epoch": row.get('epoch', 'Неизвестно'),
            "image_path": row['local_path'],
            "similarity": float(similarities[idx])
        })
        
    return similar_items

def get_aggregated_attributes(similar_items):
    """Определяет доминирующие жанр и эпоху среди похожих картин."""
    genres = [item['genre'] for item in similar_items if item['genre'] not in ['Неизвестно', 'Другое / Неизвестно']]
    epochs = [item['epoch'] for item in similar_items if item['epoch'] != 'Неизвестно']
    
    # Берем самое частое значение (моду)
    pred_genre = max(set(genres), key=genres.count) if genres else "Не определено"
    pred_epoch = max(set(epochs), key=epochs.count) if epochs else "Не определено"
    
    return pred_genre, pred_epoch