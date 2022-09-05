import numpy as np

filename = 'singleLinear02.csv'
data = np.loadtxt(filename, delimiter=',')
# print(type(data))
# print(data)

table_col = data.shape[1] # 열개수

y_column = 1 # 정답(출력) 데이터 컬럼수
x_column = table_col - y_column # 입력 데이터 컬럼수

x = data[:, 0:x_column]
y = data[:, x_column:]
#
# print('x :', x)
# print('-'*30)
#
# print('y :', y)
# print('-'*30)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=1/4, random_state=0)

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

model = Sequential() # 모델 객체 생성

from tensorflow.python.keras.layers import Dense

model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=10000, batch_size=10000, verbose=1)

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.title('회귀선과 실제 정답의 산점도 그래프')
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x_train, y_train, 'k.') # 정답을 위한 산점도

train_pred = model.predict(x_train)
plt.plot(x_train, train_pred, 'r') # 최적합 회귀선 그리기

print('가중치 정보(w, b)')
print(model.get_weights())

filename = 'singleLinear02.png'
plt.savefig(filename)

print('테스트 데이터로 예측해보기')
prediction = model.predict(x_test)

for idx in range(len(y_test)):
    label = y_test[idx] # 정답을 label이라고 부릅니다.
    pred = prediction[idx] # 테스트 용 데이터를 사용한 예측치

    print('정답 : %.4f, 예측값 : %.4f' % (label, pred))

print('finished')