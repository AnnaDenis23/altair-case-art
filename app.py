import streamlit as st
from PIL import Image
from utils.preprocessing import get_image_embedding
from utils.confidence import evaluate_confidence
from models.clip_search import find_similar, get_aggregated_attributes
from models.classifier import classify_image
from components.attribution_card import render_attribution_card

# Настройки страницы
st.set_page_config(page_title="ИИ-Искусствовед", page_icon="🎨", layout="wide")

# Кэшируем загрузку модели и данных, чтобы не грузить их при каждом обновлении
@st.cache_resource
def load_models():
    # Просто вызываем функции, чтобы они инициализировались и закэшировались
    from models.clip_search import load_data
    load_data()
    return True

def main():
    st.title("🎨 ИИ-Искусствовед: Интеллектуальный помощник")
    st.markdown("Система предварительной атрибуции произведений искусства. Загрузите изображение, чтобы получить гипотезу о его происхождении.")
    
    load_models()
    
    # Загрузка изображения
    uploaded_file = st.file_uploader("Загрузите изображение картины", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Показываем загруженную картинку
        st.image(image, caption="Загруженное произведение", width=400)
        
        if st.button("🔍 Провести анализ"):
            with st.spinner("Анализируем изображение..."):
                # 1. Получаем эмбеддинг загруженной картинки
                query_embedding = get_image_embedding(image)
                
                # ==========================================
                # ПОДХОД 1: Поиск похожих (Similarity Search)
                # ==========================================
                similar_items = find_similar(query_embedding)
                max_similarity = similar_items[0]['similarity'] if similar_items else 0.0
                
                pred_genre_search, pred_epoch_search = get_aggregated_attributes(similar_items)
                confidence_search = evaluate_confidence(max_similarity)
                
                # ==========================================
                # ПОДХОД 2: Прямая классификация (Classification)
                # ==========================================
                class_results = classify_image(query_embedding)
                
                # Для классификатора уверенность берем как среднее между жанром и эпохой
                avg_class_conf = (class_results['genre_confidence'] + class_results['epoch_confidence']) / 2
                confidence_class = evaluate_confidence(avg_class_conf)
                
                # ==========================================
                # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ (Сравнение подходов)
                # ==========================================
                st.header("📊 Результаты анализа")
                
                tab1, tab2 = st.tabs(["🔍 Подход 1: Поиск похожих (Similarity Search)", 
                                      "🎯 Подход 2: Прямая классификация (Zero-Shot)"])
                
                with tab1:
                    render_attribution_card(
                        title="Анализ через визуальный поиск",
                        predicted_genre=pred_genre_search,
                        predicted_epoch=pred_epoch_search,
                        confidence_data=confidence_search,
                        similar_items=similar_items,
                        approach_name="Гипотеза на основе поиска аналогов"
                    )
                    
                with tab2:
                    # Для второго таба тоже покажем похожие, но акцент на метриках классификатора
                    render_attribution_card(
                        title="Анализ через прямую классификацию",
                        predicted_genre=class_results['genre'],
                        predicted_epoch=class_results['epoch'],
                        confidence_data=confidence_class,
                        similar_items=similar_items, # Показываем те же похожие для контекста
                        approach_name=f"Гипотеза на основе Zero-Shot (Уверенность в жанре: {class_results['genre_confidence']:.1%}, в эпохе: {class_results['epoch_confidence']:.1%})"
                    )

if __name__ == "__main__":
    main()