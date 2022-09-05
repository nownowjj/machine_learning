import numpy as np

filename = 'multiLinear01.csv'
data = np.loadtxt(filename, delimiter=',')
# print(data.shape)
# print('-'*30)

table_col =data.shape[1] # 컬럼수
y_column = 1
x_column = table_col - y_column

# 입력 데이터와 출력 데이터를 분리하기
x = data[:, 0:x_column]
y = data[:, x_column:]

from sklearn.model_selection import train_test_split

seed = 0
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=seed)

# print('x_train :', x_train)
# print('-'*30)
#
# print('x_test :', x_test)
# print('-'*30)
#
# print('y_train :', y_train)
# print('-'*30)
#
# print('y_test :', y_test)
# print('-'*30)

from tensorflow.python.keras.models import Sequential
model = Sequential()

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=10000, batch_size=10000, verbose=1)

# import matplotlib.pyplot as plt
# plt.rc('font', family='Malgun Gothic')
#
# plt.figure()
# plt.title('회귀선과 산점도 그래프')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x_train, y_train, 'k.')
#
# train_pred = model.predict(x_train)
# plt.plot(x_train, train_pred, 'r')
#
# filename = 'multiLinear01.png'
# plt.savefig(filename)
# print(filename + ' 파일 저장됨')

print('테스트용 데이터로 예측해보기')
# 테스트 용 데이터에 대한 예측값
prediction = model.predict(x_test)

# 실제 정답과 예측 값 비교 하기
for idx in range(len(y_test)):
    label = y_test[idx]
    pred = prediction[idx]

    print('정답 : %.4f, 예측 값 : %.4f' % (label, pred) )

print('finished')










