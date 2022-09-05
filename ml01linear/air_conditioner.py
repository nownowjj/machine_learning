import numpy as np

x = [26, 28, 30, 32]
y = [148, 164, 168, 183]

mx = np.mean(x)
my = np.mean(y)

print('mean(x) :', mx)
print('mean(y) :', my)

bunmo = sum([(mx-i)**2 for i in x])
print('분모 :', bunmo)

def calc(x, mx, y, my):
    result = 0

    for i in range(len(x)):
        result += (x[i] - mx) * (y[i] - my)

    return result

bunja = calc(x, mx, y, my)
print('분자 :', bunja)

w = bunja / bunmo
b = my - (mx * w)

print('기울기 w :', w)
print('절편 b :', b)