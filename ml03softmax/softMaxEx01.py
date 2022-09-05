import numpy as np

filename = 'softMaxEx01.csv'

import pandas as pd
df = pd.read_csv(filename, header=None, names=['x1', 'x2', 'x3', 'x4', 'y'])
print(df.head())
print('-'*30)

print(df['y'].unique())
print('-'*30)

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='y')
filename = 'softMaxEx01_01.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

data = df.values # dataframe --> numpy

# 데이터 분리 작업
table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column:]

from sklearn.model_selection import train_test_split

seed = 1234
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=seed)

# one hot encoding 작업
print('before one hot encoding')
print(y_train)

nb_classes = 3

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes=nb_classes, dtype='float32')

print('after one hot encoding')
print(y_train)

# 모델을 생성하고 훈련시키기
from tensorflow.python.keras.models import Sequential
model = Sequential()

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=nb_classes, input_shape=(x_column,), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000, validation_split=0.3)

# 훈련용 비용 함수와 검증용 비용 함수에 대한 시각화

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.figure()

plt.plot(history.history['loss'], color='b', linestyle='dashed', linewidth=1, label='loss')
plt.plot(history.history['val_loss'], color='r', linestyle='dashed', linewidth=1, label='val_loss')

plt.xlabel('epochs에 따른 손실 정보')
plt.legend()

filename = 'softMaxEx01_02.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

# csv 파일 만들기
csvDataList = [] # csv로 저장할 리스트
hit = 0.0 # 데이터를 맞춘 횟수

for idx in range(len(x_test)):
    H = model.predict(np.array([x_test[idx]])) # 가설 확률 값
    prediction = np.argmax(H, axis=-1) # 예측 값

    sublist = [] # 엑셀에 들어갈 1줄 정보
    sublist.append(prediction[0])
    sublist.append(int(y_test[idx]))

    _H = H.flatten() # 1차원화

    for aaa in range(len(_H)):
        sublist.append(_H[aaa])
    csvDataList.append(sublist)

    hit += float(prediction[0] == int(y_test[idx]))
# end for

hitrate = 100 * hit / len(x_test)
print('정확도 : %.4f' % (hitrate))

mycolumn = ['예측값', '정답', '확률01', '확률02', '확률03']
df = pd.DataFrame(csvDataList, columns=mycolumn)
csvFileName = 'softMaxCsv.csv'
df.to_csv(csvFileName, index=False, encoding='UTF-8')
print(csvFileName + ' 파일 저장됨')












