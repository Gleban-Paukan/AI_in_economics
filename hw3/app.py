import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from model import Predictor

# Define class names for Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

st.set_page_config(page_title="Модель классификации одежды", page_icon="👕", layout="wide")

st.title("👗 Fashion MNIST Predictor Pro")
st.markdown("Загрузите изображение предмета одежды, и модель предскажет его класс. Воспользуйтесь расширенным функционалом: интерактивной диаграммой вероятностей, тепловой картой признаков и инструментами предобработки.")

st.sidebar.header("⚙️ Параметры модели")
model_path = st.sidebar.text_input("Путь к весам модели (.pth)", value="model_weights.pth")
in_features = st.sidebar.number_input("In Features (Входные признаки)", min_value=1, value=784)
n_classes = st.sidebar.number_input("Количество классов", min_value=1, value=10)
in_channels = st.sidebar.number_input("In Channels (Каналы изображения)", min_value=1, value=1)

# Feature 3: Image Preprocessing tools 
st.sidebar.header("🛠️ Предобработка изображения")
st.sidebar.markdown("Fashion MNIST обучен на картинках с черным фоном и белой одеждой. Если вы загружаете картинку на белом фоне — включите инверсию.")
invert_colors = st.sidebar.checkbox("Инвертировать цвета", value=True)
contrast_factor = st.sidebar.slider("Контрастность", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

uploaded_file = st.file_uploader("Выберите изображение...", type=["png", "jpg", "jpeg"])

if st.button("Рассчитать", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("Пожалуйста, загрузите изображение для классификации.")
    elif not os.path.exists(model_path):
        st.error(f"Ошибка: Файл с весами модели не найден по пути '{model_path}'. Проверьте правильность пути.")
    else:
        with st.spinner("Модель анализирует изображение..."):
            try:
                # Load and display the image
                original_image = Image.open(uploaded_file)
                image = original_image.convert("L")  # Convert to grayscale
                
                # Apply preprocessing mapping
                if invert_colors:
                    image = ImageOps.invert(image)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast_factor)
                
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(original_image, caption="Оригинальное изображение", use_container_width=True)
                with col_img2:
                    st.image(image, caption="После предобработки (вид для модели)", use_container_width=True)
                
                # Transform to tensor (1, 28, 28) with values in range [0, 1]
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                ])
                
                img_tensor = transform(image).unsqueeze(0)  # Add batch dimension -> [1, 1, 28, 28]

                # Initialize the predictor
                predictor = Predictor(
                    model_path=model_path,
                    in_features=in_features,
                    n_classes=n_classes,
                    in_channels=in_channels
                )
                
                # Run inference
                pred_class_idx, confidence, all_probs = predictor.predict(img_tensor)
                
                # Extract bounds and values
                idx = int(pred_class_idx[0])
                conf = float(confidence[0])
                probs = all_probs[0]
                
                class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Неизвестный класс ({idx})"
                
                st.success("Успешная классификация!")
                
                # Display Results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="🏆 Предсказанный класс", value=class_name)
                with res_col2:
                    st.metric(label="🎯 Уверенность сети", value=f"{conf:.2%}")
                    
                st.markdown("---")
                
                col_feat1, col_feat2 = st.columns(2)
                
                with col_feat1:
                    # Feature 1: Interactive Bar Chart
                    st.subheader("📊 Распределение вероятностей")
                    prob_data = {
                        "Классы": CLASS_NAMES,
                        "Вероятность": probs
                    }
                    fig = px.bar(prob_data, x="Вероятность", y="Классы", orientation='h', color="Вероятность",
                                 color_continuous_scale="Viridis",
                                 title="Уверенность модели по каждому классу")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_feat2:
                    # Feature 2: Explainability Saliency Map
                    st.subheader("🧠 Saliency Map (Важность пикселей)")
                    st.markdown("Показывает области, на которые модель обратила наибольшее внимание при предсказании.")
                    saliency_map = predictor.get_saliency_map(img_tensor, idx)
                    
                    fig_saliency = px.imshow(saliency_map, color_continuous_scale="inferno")
                    fig_saliency.update_xaxes(showticklabels=False)
                    fig_saliency.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_saliency, use_container_width=True)
                
            except Exception as e:
                st.error(f"Произошла ошибка в процессе предсказания: {str(e)}")
