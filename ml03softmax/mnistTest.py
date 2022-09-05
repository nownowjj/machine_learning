from pandas import Series

import matplotlib.pyplot as plt

data = [0.9218, 0.926, 0.9737, 0.9746, 0.9752]
myseries =Series(data)
myseries.plot(kind='bar')

filename = 'mnist_test_result.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')