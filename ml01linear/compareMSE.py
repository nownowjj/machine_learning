import numpy as np

x = np.array([26, 28, 30, 32])
y = np.array([148, 164, 168, 183])

w = np.arange(0.0, 10.91, 0.001)

def myfunction(w, x):
    data = w * x + 7.7
    return np.sum((data - y)**2)

loss = []
for some in w :
    data = myfunction(some, x)
    loss.append(data)

print('오차 함수')
print(loss)

minlost = min(loss)


import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.plot(w, loss, color='g', linestyle='solid', linewidth=1, label='이차 곡선')

plt.plot(5.45, minlost, color='r', linestyle='solid', marker='o')

plt.grid(True)
filename = 'compareMSE.png'
plt.savefig(filename)
print(filename + ' 파일 저장')

