import streamlit as st
from PIL import Image
import os

def render_attribution_card(title, predicted_genre, predicted_epoch, 
                            confidence_data, similar_items, approach_name):
    """Рендерит карточку атрибуции в Streamlit."""
    
    st.subheader(f"🔬 {approach_name}")
    
    # 1. Блок с основными предсказаниями
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Предполагаемый жанр", predicted_genre)
    with col2:
        st.metric("Предполагаемая эпоха", predicted_epoch)
    with col3:
        st.metric("Уверенность модели", f"{confidence_data['level']}")
        
    # 2. Флаг экспертной проверки
    if confidence_data['needs_expert']:
        st.error(confidence_data['message'])
    else:
        st.success(confidence_data['message'])
        
    st.divider()
    
    # 3. Блок с похожими работами
    st.write("**Визуально похожие работы из коллекции:**")
    
    cols = st.columns(len(similar_items))
    for i, col in enumerate(cols):
        with col:
            item = similar_items[i]
            if os.path.exists(item['image_path']):
                st.image(item['image_path'], use_column_width=True)
            else:
                st.warning("Изображение не найдено")
            
            st.caption(f"**{item['title']}**")
            st.caption(f"_{item['author']}_")
            st.caption(f"Жанр: {item['genre']} | Эпоха: {item['epoch']}")
            st.caption(f"📊 Сходство: {item['similarity']:.2%}")