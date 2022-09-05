from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train.shape :', x_train.shape)
print('x_train.ndim :', x_train.ndim)
print('x_train.dtype :', x_train.dtype)

print('len(y_train) :', len(y_train))
print('y_train :', y_train)

print('x_test.shape :', x_test.shape)
print('x_test.ndim :', x_test.ndim)
print('x_test.dtype :', x_test.dtype)

print('len(y_test) :', len(y_test))
print('y_test :', y_test)

digit = x_train[2]

import matplotlib.pyplot as plt

plt.imshow((digit))
filename = "mnistinfo.png"
plt.savefig(filename)
print(filename + ' 파일 저장됨')