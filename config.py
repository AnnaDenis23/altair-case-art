import os

# === Базовые пути ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка с картинками
IMAGES_DIR = os.path.join(BASE_DIR, "data", "jpg")

# Папка с файлами данных (csv и npy)
DATA_DIR = os.path.join(BASE_DIR, "files")
METADATA_PATH = os.path.join(DATA_DIR, "russian_paintings_processed.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "paintings_embeddings.npy")

# === Настройки модели ===
CLIP_MODEL_NAME = "ViT-B/32"

# === Пороги уверенности (для кейса) ===
# Если сходство (cosine similarity) ниже этого порога, модель "сомневается"
SIMILARITY_THRESHOLD_HIGH = 0.70  # Высокая уверенность
SIMILARITY_THRESHOLD_LOW = 0.35  # Низкая уверенность -> нужен эксперт

# === Параметры поиска ===
TOP_K_SIMILAR = 5  # Сколько похожих картин показывать в карточке