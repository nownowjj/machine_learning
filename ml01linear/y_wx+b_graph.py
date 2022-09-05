import numpy as np

def myfunction(x):
    return 5.45 * x + 7.7

x = np.arange(26.0, 32.1, 2.0)
print(x)

y_answer = [148.0, 164.0, 168.0, 183.0]

y = myfunction(x)

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.plot(x, y_answer, marker='o', color='g', linestyle='none', label='label')

plt.plot(x, y, marker='', color='r', linestyle='solid', label='이상적인 직선')

plt.legend(loc='upper left')
plt.grid(True)

filename = 'figure01.png'
plt.savefig(filename)
print(filename + ' 파일 저장')


