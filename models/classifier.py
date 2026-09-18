import torch
import clip
from utils.preprocessing import clip_model, device

# Текстовые промпты для Zero-Shot классификации
GENRE_PROMPTS = [
    "a portrait painting of a person",
    "a landscape painting with nature",
    "a still life painting with objects",
    "a historical painting with events",
    "a genre painting of daily life"
]
GENRE_LABELS = ["Портрет", "Пейзаж", "Натюрморт", "Историческая живопись", "Бытовой жанр"]

EPOCH_PROMPTS = [
    "an 18th century classic painting",
    "a 19th century classic painting",
    "a 20th century classic painting"
]
EPOCH_LABELS = ["XVIII век", "XIX век", "XX век"]

def classify_image(image_embedding):
    """
    Классифицирует изображение по жанру и эпохе, сравнивая его эмбеддинг 
    с текстовыми эмбеддингами промптов (Zero-Shot).
    """
    with torch.no_grad():
        # 1. Классификация жанра
        text_tokens = clip.tokenize(GENRE_PROMPTS).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Считаем сходство с текстами
        logits_genre = (image_embedding @ text_features.T).softmax(dim=-1)
        genre_probs = logits_genre.cpu().numpy()[0]
        
        best_genre_idx = genre_probs.argmax()
        pred_genre = GENRE_LABELS[best_genre_idx]
        genre_confidence = float(genre_probs[best_genre_idx])
        
        # 2. Классификация эпохи
        text_tokens_epoch = clip.tokenize(EPOCH_PROMPTS).to(device)
        text_features_epoch = clip_model.encode_text(text_tokens_epoch)
        text_features_epoch = text_features_epoch / text_features_epoch.norm(dim=-1, keepdim=True)
        
        logits_epoch = (image_embedding @ text_features_epoch.T).softmax(dim=-1)
        epoch_probs = logits_epoch.cpu().numpy()[0]
        
        best_epoch_idx = epoch_probs.argmax()
        pred_epoch = EPOCH_LABELS[best_epoch_idx]
        epoch_confidence = float(epoch_probs[best_epoch_idx])

    return {
        "genre": pred_genre,
        "genre_confidence": genre_confidence,
        "epoch": pred_epoch,
        "epoch_confidence": epoch_confidence
    }