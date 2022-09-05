menu = int(input('콜백 함수 사용 여부 => no(0), yes(1) : '))

filename = 'zoo.data.txt'

import pandas as pd
df=pd.read_csv(filename, index_col='name')
# print(df.head())
# print('-'*30)

print(df.columns)
print('-'*30)

corr = df.corr() # 상관 계수를 구해주는 함수
print(corr)
print('-'*30)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
sns.heatmap(corr, linewidth=1, vmax=0.5, linecolor='white', annot=True)

filename = 'data_zoo_corr_image.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

# 비교적 상관 관계 계수가 큰 항목들만 따로 만들어서 시각화
pair_df = df[['eggs', 'milk', 'backbone', 'venomous', 'type']]
sns.pairplot(pair_df, hue='type')

filename = 'data_zoo_pair_plot.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

data = df.values

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]

y_raw = data[:, x_column:].ravel()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_raw)

y = le.transform(y_raw)

print(y)
print('-'*30)

x=x.astype(float)
y=y.astype(float)

from sklearn.model_selection import train_test_split

seed = 1234
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=seed)

nb_classes = 7
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes=nb_classes, dtype='float32')

print(x_train)
print('-'*30)

print(y_train)
print('-'*30)

# 모델을 준비하고 학습하기
from tensorflow.python.keras.models import Sequential
model = Sequential()

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=nb_classes, input_shape=(x_column,), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('데이터로 학습 중입니다. 잠시만 기다려 주세요.')
if menu == 0:
    history = model.fit(x_train, y_train, epochs=10000, validation_split=0.3)
else :
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=500)
    history = model.fit(x_train, y_train, epochs=10000, validation_split=0.3, callbacks=[es])
# end if

print(history)

# 정확도와 손실 함수에 대한 시각화 작업
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure()
plt.plot(accuracy, 'b--', label='training accuracy')
plt.plot(val_accuracy, 'r--', label='validation accuracy')

plt.title('epoch에 따른 정확도 그래프')
plt.legend()

if menu == 0:
    filename = 'data_zoo_figure_01(no).png'
else :
    filename = 'data_zoo_figure_01(yes).png'
# end if

plt.savefig(filename)
print(filename + ' 파일 저장됨')

print('손실 함수에 대한 그래프')
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(loss, 'b--', label='training loss')
plt.plot(val_loss, 'r--', label='validation loss')

plt.title('epoch에 따른 손실 함수 그래프')
plt.legend()

if menu == 0:
    filename = 'data_zoo_figure_02(no).png'
else :
    filename = 'data_zoo_figure_02(yes).png'
# end if

plt.savefig(filename)
print(filename + ' 파일 저장됨')

# csv 파일 만들기
csvDataList = [] # csv로 저장할 리스트
hit = 0.0 # 데이터를 맞춘 횟수

import numpy as np

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

mycolumn = ['예측값', '정답', '확률값0', '확률값1', '확률값2', '확률값3', '확률값4', '확률값5', '확률값6']
df = pd.DataFrame(csvDataList, columns=mycolumn)
csvFileName = 'data_zoo_excel_csv.csv'
df.to_csv(csvFileName, index=False, encoding='UTF-8')
print(csvFileName + ' 파일 저장됨')





