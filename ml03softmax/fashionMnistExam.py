from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 클래스 각 품목의 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 이미지 1개를 파일로 저장
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(x_train[1])
plt.grid(False)
filename = 'fashionMnistExam_01.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')

# 데이터에 대한 정규화
x_train, x_test = x_train.astype(float)/255.0, x_test.astype(float)/255.0

num_rows = 5 # 행수
num_cols = 5 # 열수
num_images = num_rows * num_cols # 그리고자 하는 이미지 갯수
plt.figure(figsize=(10, 10))

for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
# end for

filename = 'fashionMnistExam_02.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')

nb_classes = 10 # 이미지 종류 10개

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=nb_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=0)

# 모델에 대하여 정확도 및 손실을 평가합니다.
score = model.evaluate(x_test, y_test, verbose=2)

print('test accuracy : %.4f' % (score[1]))
print('test loss : %.4f' % (score[0]))

# 예측하기
prediction = model.predict(x_test)

check_length = 5 # 예측해볼 데이터의 개수
print('테스트 이미지 ' + str(check_length) + '개 예측해보기')
print('예측 확률')
print(prediction[0:check_length])

import numpy as np

print('예측 이미지')
print(np.argmax(prediction[0:check_length], axis=-1))

print('정답 이미지')
print(y_test[0:check_length])


def plot_image(i, prediction_array, true_label, img):
    # i : 이미지 색인 번호
    # prediction_array : 예측 확률을 저장하고 있는 배열
    # true_label : 실제 정답 데이터
    # img : 테스트를 수행하기 위한 이미지 배열
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary) # draw image

    prediction_label = np.argmax(prediction_array[i]) # 예측 값
    if prediction_label == true_label[i] : # 정답을 맞춘 경우 파란색
        mycolor = 'blue'
    else :
        mycolor = 'red'

    pred = class_names[prediction_label] # 예측값
    prob = 100.0 * np.max(prediction_array[i]) # 제일 큰 확률값
    answer = class_names[true_label[i]]
    plt.xlabel("{} {:6.2f}% ({})".format(pred, prob, answer), color=mycolor)
# end plot_image

def plot_value_array(prediction_array, true_label):
    # prediction_array : 예측 확률을 저장하고 있는 배열
    # true_label : 실제 정답 데이터
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0.0, 1.0])
    thisplot = plt.bar(range(10), prediction_array, color="#777777")

    prediction_label = np.argmax(prediction_array)
    thisplot[prediction_label].set_color('red') # 예측이 틀림
    thisplot[true_label].set_color('blue') # 예측함
# end def plot_value_array

# 0번째 test 데이터를 이용하여 이미지와 확률 값에 대한 그래프
idx = 0 # 이미지 색인 번호
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(idx, prediction, y_test, x_test)

plt.subplot(1, 2, 2)
plot_value_array(prediction[idx], y_test[idx])

filename = 'fashionMnistExam_03.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for idx in range(num_images) :
    plt.subplot(num_rows, 2*num_cols, 2*idx+1)
    plot_image(idx, prediction, y_test, x_test)

    plt.subplot(num_rows, 2 * num_cols, 2*idx+2)
    plot_value_array(prediction[idx], y_test[idx])
# end for

filename = 'fashionMnistExam_04.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')











