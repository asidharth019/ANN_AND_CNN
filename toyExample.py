import numpy as np

X = [0.5, 2.5]
Y = [0.2, 0.9]

def grdw(w,b,x,y):
    fx = 1.0 / (1.0 + np.exp(-(w*x +b)))
    return (fx-y)*fx*(1-fx)*x

def grdb(w,b,x,y):
    fx = 1.0 / (1.0 + np.exp(-(w*x +b)))
    return (fx-y)*fx*(1-fx)

w, b, eta, = -2,-2,1.0
for i in range(0,1000):
    dw, db = 0, 0
    for x, y in zip(X,Y):
        dw += grdw(w,b,x,y)
        db += grdb(w,b,x,y)
    w = w - eta*dw
    b = b - eta*db

print(w)
print(b)