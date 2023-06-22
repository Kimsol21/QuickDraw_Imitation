import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

'''
'''

# 이미지 경로 설정
mermaid_folder = './data/mermaid'  # 인어 이미지 폴더 경로
panda_folder = './data/panda'  # 판다 이미지 폴더 경로

# 이미지 크기 설정
image_size = (32, 32) # 이미지의 너비와 높이가 각각 32픽셀.

# 이미지 데이터와 레이블을 저장할 리스트
data = []
labels = []

# 인어 이미지 데이터 읽어오기
for image_file in os.listdir(mermaid_folder): # mermaid_folder에 있는 파일을 하나씩 읽어옴.
    if len(data) >= 1200: # 이미지 데이터가 1200개 이상인 경우 루프 종료.
        break
    image_path = os.path.join(mermaid_folder, image_file) # 이미지파일의 경로 변수에 저장.
    image = cv2.imread(image_path) # 앞서 저장한 경로를 이용해 이미지 파일 읽어옴.
    image = cv2.resize(image, image_size) # 이미지 크기를 위에서 정한 크기로 재설정.
    image = img_to_array(image) # 이미지를 배열로 변환.
    data.append(image) # 이미지 데이터를 data 리스트에 추가.
    labels.append(0)  # 해당 이미지의 레이블을 0으로 설정. 인어는 0으로 레이블링

# 판다 이미지 데이터 읽어오기, 위와 동일한 과정.
for image_file in os.listdir(panda_folder):
    if len(data) >= 2400:
        break
    image_path = os.path.join(panda_folder, image_file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = img_to_array(image)
    data.append(image)
    labels.append(1)  # 판다는 1로 레이블링

# 데이터 전처리
# 데이터, 라벨 리스트를 넘파이 배열로 변환.
# 데이터 타입은 float32로 설정, 값의 범위를 0부터 1사이로 정규화하기 위해 255로 나눔.
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)

# 데이터 분할(학습 세트, 테스트 세트)
# 20%가 테스트 세트로 사용.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 모델 정의 및 컴파일
model = tf.keras.Sequential([ # Sequential 모델을 생성.
    # 2D 컨볼루션 수행하는 레이어. 32개의 필터 사용, 각 필터의 크기는 (3,3),
    # 활성화 함수로는 'relu'사용, 입력 이미지의 형태는 (32,32,3)이다.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

    # 최대 풀링을 수행하는 레이어. (2,2)크기의 풀링 필터 사용.
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 다차원 입력을 1차원으로 평탄화하는 역할.
    tf.keras.layers.Flatten(),
    #완전 연결층. 64개의 뉴런을 가지고, 'relu'활성화 함수 사용.
    tf.keras.layers.Dense(64, activation='relu'),
    #완전 연결층. 이진 분류를 위한 출력층. 하나의 뉴런을 가지고, 'sigmoid'활성화 함수 사용.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#모델을 컴파일.
model.compile(optimizer='adam',#adam 옵티마이저
              loss=tf.keras.losses.BinaryCrossentropy(),#loss로는 이진 분류를 위한 이진 크로스엔트.
              metrics=['accuracy'])

# 모델 학습 함수 : 학습 중에는 손실 함수 값과 정확도가 출력, 검증 데이터를 통해 모델의 성능을 평가.
# 학습 데이터의 입력값, 해당 입력값에 대한 정답 레이블
# 전체 데이터셋을 반복하여 학습하는 횟수.
# 한 번의 학습 단계에서 사용되는 샘플의 개수. 모델 가중치 업데이트는 배치 단위로 이루어짐.
# 검증 데이터 지정.
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 손실 그래프 그리기
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.title('loss 그래프')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# 모델 저장
model.save('./quickdraw_model.h5')

'''  
'''


'''

#모델이 잘 동작하는 지 테스트
# 이미지 경로 설정
My_image_path = './MyDrawing.jpg'  # 내가 그린 이미지 파일 경로

# 이미지 크기 설정
image_size = (32, 32)

# 이미지 데이터 전처리
image = cv2.imread(My_image_path)
image = cv2.resize(image, image_size)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image.astype('float32') / 255.0

# 모델 로드
model = tf.keras.models.load_model('./quickdraw_model.h5')  # 학습된 모델 파일 경로

# 이미지 분류
prediction = model.predict(image) # 내가 그린 이미지에 대한 예측 수행.
class_label = 'Mermaid' if prediction[0] < 0.5 else 'Panda' # 예측 결과의 첫번째 요소 : 0과 1사이의 확률.
# 만약 확률이 0.5보다 낮으면 인어, 그렇지 않으면 판다로 판단.
print(f"The image is classified as: {class_label}") # 예측 결과 출력.


'''