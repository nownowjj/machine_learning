filename='iris.csv'

import pandas as pd
df=pd.read_csv(filename)
print(df.head())

# 정답의 종류가 3개인 다중 분류 입니다.
label='Species'
print(df[label].unique())

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue=label)
imagefilename='irisSoftMax01_01.png'
plt.savefig(imagefilename)
print(imagefilename + ' 파일 저장됨')

data=df.values # dataframe → numpy array

# 학습을 진행하기 위한 기초 변수들을 정의합니다.
table_col=data.shape[1]
y_column=1
x_column=table_col-y_column

x=data[:, 0:x_column]
y_raw=data[:, x_column:].ravel()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(y_raw)
y=le.transform(y_raw)

x=x.astype(float)
y=y.astype(float)

from sklearn.model_selection import train_test_split

seed=1234
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=seed)

# one hot encoding 작업을 합니다.
# print('before one hot encoding')
# print(y_train)

nb_classes=3 # 정답이 될 수 있는 갯수(클래스의 갯수)

from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train, num_classes=nb_classes, dtype='float32')

# print('after one hot encoding')
# print(y_train)

# 모델을 생성하고, 훈련용 데이터를 학습시킵니다.
from tensorflow.python.keras.models import Sequential
model=Sequential()

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=nb_classes, input_shape=(x_column,), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=1000, verbose=0, validation_split=0.3)
print(history)

# 손실 함수에 대한 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure()

plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')

plt.xlabel('epoch에 따른 손실 정보')
plt.legend()
imagefilename='irisSoftMax01_02.png'
plt.savefig(imagefilename)
print(imagefilename + ' 파일 저장됨')

csvDataList=[] # csv 파일로 저장될 리스트
hit=0.0 # 데이터를 맟춘 개수

import numpy as np

for idx in range(len(x_test)):
    # idx 번째 테스트 데이터를 이용하여 예측을 해봅니다.
    H=model.predict(np.array([x_test[idx]]))
    # print(H) # H는 각 class가 가지고 있는 확률 정보를 출력해 줍니다.
    # 예시) [0.7706099  0.16933888 0.06005123]
    prediction=np.argmax(H, axis=-1)
    print('\n예측 값 :', prediction, end=' ')
    print('정답 : [%d]' % int(y_test[idx]), end=' ')

    sublist=[] # 엑셀에 들어갈 한줄 정보(예측값, 정답, 확률값01,  확률값02,  확률값03)
    sublist.append(prediction[0])
    sublist.append(int(y_test[idx]))

    _H = H.flatten() # 1차원화

    for cnt in range(len(_H)):
        sublist.append(_H[cnt])

    csvDataList.append(sublist)

    # 비교하면 참/거짓을 실수화하면 1.0/0.0이 됩니다.
    hit += float(prediction[0] == int(y_test[idx]))

    # if idx == 1 :
    #     break
# end for

hitrate=100*hit/len(x_test) # 정확도
print('\n정확도 : %.4f' % (hitrate))

mycolumns=['예측값', '정답', '확률01', '확률02', '확률03']
df=pd.DataFrame(csvDataList, columns=mycolumns)
csvfilename='irisSoftMaxCsv.csv'
df.to_csv(csvfilename, index=False, encoding='cp949')
print(csvfilename + ' 파일이 저장됨')

print('finished')
