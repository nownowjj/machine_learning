from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_column = 28*28
x_train = x_train.reshape((60000, x_column)) # 형상 변경
x_train = x_train.astype(float)/255

x_test = x_test.reshape((10000, x_column))
x_test = x_test.astype(float)/255

print('before y_train[0]')
print(y_train[0])

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print('after y_train[0]')
print(y_train[0])

# 모델 생성후 학습을 진행
from tensorflow.python.keras.models import Sequential
model = Sequential()

nb_classes = 10

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=nb_classes, input_shape=(x_column,), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('model.fit 진행중입니다.')
history = model.fit(x_train, y_train, validation_split=0.3, epochs=5, batch_size=64)

print('history의 모든 데이터 목록보기')
print(history.history.keys())

print('model을 평가합니다.')
score = model.evaluate(x_test, y_test, verbose=1)
print(score)
print('-'*30)

print('평가 지표 목록')
print(model.metrics_names)
print('-'*30)

print('test loss : %.4f' % (score[0]))
print('test accuracy : %.4f' % (score[1]))

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

# 정확도에 대한 데이터 시각화
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')

filename = 'mnistNeuralNet01_01.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')


# 손실 함수에 대한 데이터 시각화
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')

filename = 'mnistNeuralNet01_02.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')