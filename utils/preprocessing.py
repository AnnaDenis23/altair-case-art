import os
import torch
import clip
from PIL import Image
import config

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# ======================================

# Загружаем модель CLIP один раз
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)

# Загружаем модель CLIP один раз
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)

def resolve_image_path(csv_filename: str) -> str:
    """
    Берет имя файла из CSV (например, './data/845010375.jpg') 
    и возвращает правильный путь в локальной папке data/jpg/
    """
    filename = os.path.basename(csv_filename)
    return os.path.join(config.IMAGES_DIR, filename)

def get_image_embedding(image: Image.Image) -> torch.Tensor:
    """Превращает изображение PIL в нормализованный вектор (эмбеддинг)."""
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        # Нормализация для косинусного сходства
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features