import numpy as np
import matplotlib.pyplot as plt

def grad_step_init():
    global curX
    global newX
    global deltaX
    global xlog
    global ylog
    
    newX = np.uint8(newX + .5)
    for i in range(dimensie-1):
        y0[i] = Y[curX[i]]
    np.append(xlog, newX)
    np.append(ylog, Y[newX])
    for i in range(dimensie-1):
        y1[i] = Y[np.uint8(newX[i])]
        deltaX[i]= learningRate*(y1[i] + y0[i])*np.sign(deltaX[i])
    newX = curX + deltaX
    #curX = np.uint8(newX + .5)
    xlog = np.append(xlog, curX)
    ylog = np.append(ylog, Y[curX])

def grad_step():
    global curX
    global newX
    global deltaX
    global y1
    global xlog
    global ylog
    
    for i in range(dimensie-1):
        newX[i] = curX[i] + deltaX[i]
    newX = np.uint8(newX + .5)
    y0 = y1
    y1 = Y[newX]
    deltaX = learningRate*(y1 - y0)*np.sign(deltaX) + alpha*deltaX
    #curX = np.uint8(newX + .5)
    curX = newX
    xlog = np.append(xlog, curX)
    ylog = np.append(ylog, Y[curX])

dimensie = 100
deltaX = np.array([20]*dimensie)
curX = np.random.randint(256, size = dimensie)
newX = curX + deltaX

y0 = np.zeros(dimensie)
y1 = np.zeros(dimensie)

learningRate = 50
alpha = 0.4
xlog = np.array([])
ylog = np.array([])

X = np.arange(0, 256, 1)
Y = np.sin(X*6/255+5.2) + np.random.rand(256)*0.5

xlog = np.append(xlog, curX)
ylog = np.append(ylog, Y[curX])

grad_step_init()

for i in range(10):
    grad_step()

plt.axhline(y=0, color='k')
plt.plot(X, Y)
#plt.plot(xlog, ylog, c='red')
#plt.plot((xlog[-1],xlog[-1]), (0,ylog[-1]), c='red', zorder=10,dashes=[4, 4])
#plt.show()
    
#plt.contour(X,Y,Y)
for p in range(dimensie):
    plt.plot(xlog[p::dimensie], ylog[p::dimensie])
    plt.plot((xlog[p::dimensie][-1],xlog[p::dimensie][-1]), (0, ylog[p::dimensie][-1]), c='red', zorder=10,dashes=[4, 4])
#plt.plot(np.transpose([xlog[1::dimensie],ylog[1::dimensie]]))
plt.show()