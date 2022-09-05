import numpy as np

filename = 'surgeryTest.csv'
data = np.loadtxt(filename, delimiter=',')
# print(data.shape)
# print('-'*30)

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column:]

# print(x.shape)
# print('-'*30)
#
# print(y.shape)
# print('-'*30)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2)

print('y_test')
print(y_test)
print('-'*30)

from tensorflow.python.keras.models import Sequential
model = Sequential()

from tensorflow.python.keras.layers import Dense


model.add(Dense(units=30, input_dim=x_column, activation='relu'))

model.add(Dense(units=y_column, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=0)

# print(model.metrics_names)

score = model.evaluate(x_train, y_train)
print('train loss : %.4f' % score[0])
print('-'*30)

print('train accuracy : %.4f' % score[1])
print('-'*30)

pred = model.predict(x_test)
# print(pred)
# print('-'*30)

prediction = np.argmax(pred, axis=-1)
# print(prediction)
# print('-'*30)

print('실제 정답과 예측값 동시 출력')
for idx in range(len(prediction)):
    label = y_test[idx]
    print('real : %.f, prediction : %.f' % (label, prediction[idx]))

print('-'*30)














