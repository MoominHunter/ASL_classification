# 3.10 버전이하에서만 작동합니다.
# conda create -n test2 python=3.10
from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = [line.strip() for line in open('labels.txt', 'r').readlines()]

# Streamlit UI
st.title("🤟 수어 판별기 ✨")
st.markdown("### 수화를 인식하여 어떤 의미인지 판별해드립니다! 😊")

# 선택 옵션: 카메라 입력 또는 파일 업로드
st.sidebar.header("📷 입력 방식 선택")
input_method = st.sidebar.radio("이미지 입력 방식", ["카메라 사용", "파일 업로드"])

# 이미지 입력
if input_method == "카메라 사용":
    img_file_buffer = st.camera_input("수어를 표현해보세요 ✋")
else:
    img_file_buffer = st.file_uploader("🖼️ 이미지 파일 업로드", type=["png", "jpg", "jpeg"])

# 빈 배열 생성 (224x224x3)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if img_file_buffer is not None:
    # 이미지 열기 및 RGB 변환
    image = Image.open(img_file_buffer).convert('RGB')

    # 이미지 리사이징 (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 넘파이 배열 변환 및 정규화
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # 데이터를 모델 입력 형식에 맞게 저장
    data[0] = normalized_image_array

    # 모델 예측 수행
    prediction = model.predict(data)

    # 가장 높은 확률을 가진 클래스 예측
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 예측 결과 표시
    st.subheader("✅ 판별 결과")
    st.success(f"🔹 인식된 수어: **{class_name}**")
    st.write(f"📊 예측 정확도: **{confidence_score:.2%}**")

    # 신뢰도 프로그레스 바 추가
    st.progress(float(confidence_score))

    # 예측 이미지 미리보기
    st.image(image, caption="📷 입력된 이미지", use_column_width=True)
