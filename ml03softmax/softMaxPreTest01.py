x = [2.0, 1.0, 0.1]
y = [] # 지수 함수의 연산 결과가 저장됨

import  numpy as np

for idx in range(len(x)):
    y.append(np.exp(x[idx]))

mysum = np.sum(y)
print('총합 : ', mysum)

prop_total = 0.0
prop_data = [] # 각 확률을 저장할 리스트
for idx in range(len(y)):
    prop = y[idx]/mysum
    prop_total += prop
    prop_data.append(prop)

print(prop_data)
print('-'*30)

print(prop_total)
print('-'*30)

arr = np.array(x)

maxidx = np.argmax(arr)
print('maxidx : ', maxidx)

import matplotlib.pyplot as plt
plt.pie(prop_data, labels=prop_data)
filename = 'softMaxPreTest01.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')












