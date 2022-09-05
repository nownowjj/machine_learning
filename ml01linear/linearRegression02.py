import pandas as pd

filename = 'manhattan.csv'
data = pd.read_csv(filename)

# print('파일 기본 정보')
# print(data.info())
# print('-'*30)
#
# print('파일의 컬럼 정보')
# print(data.columns)
# print('-'*30)

# 입력과 출력 데이터를 분리합니다.
x = data[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = data[['rent']]

from sklearn.model_selection import train_test_split

SEED = 1234
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=SEED)

from sklearn.linear_model import LinearRegression
model = LinearRegression() # 모델 객체 생성

model.fit(x_train, y_train) # 훈련 데이터로 학습(피팅)

y_predict = model.predict(x_test) # 테스트 데이터로 예측하기

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.figure()
plt.scatter(y_test, y_predict, alpha=0.6)
plt.xlabel('정답 데이터')
plt.ylabel('예측 데이터')
plt.title('다중 선형 회귀')
plt.xlim([0, 20000])
plt.ylim([0, 20000])
filename = 'linearRegression02_01.png'
plt.savefig(filename)
print(filename + ' 파일 저장')


print('학습(fit) 이후의 회귀 계수 구하기')
print('기울기 :', model.coef_)
print('절편 :', model.intercept_)


plt.figure()
plt.scatter(data[['size_sqft']], data[['rent']], alpha=0.6)
plt.xlabel('주택 면적')
plt.ylabel('임대료')
plt.title('주택 면적과 임대료의 산점도')
plt.xlim([0, 20000])
plt.ylim([0, 20000])
filename = 'linearRegression02_02.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

prediction = model.predict(x_test)

print('score 함수는 결정 계수를 구해주는 함수')
print('결정 계수 : %.3f' % model.score(x_test, y_test))

print('finished')







