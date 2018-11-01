__author__ = 'Julian'
#creacion de una  maquina de apredizaje
#Fecha 14/04/2015
#prototipo1
import numpy as np
import math
#esta funcion permite crear la red de aprendizaje

def creaMLP (num_ent, num_sal, num_neu):
    W=np.random.rand(num_neu,(num_ent+1))
    V=np.random.rand(num_sal,(num_neu+1))
    return W, V

#calcula la regra de aprendizaje en este caso la
#tangente hiperbolica con la funcion math

def sigmoid(X):
    Z=np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j]= (math.tanh(X[i][j]))
    return Z

def sch (X):
    Z=np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i][j]= 1/(math.tanh(X[i][j]))
    return Z

def vmat (X):
    return  np.transpose(np.atleast_2d(X))


def operaMLP(W,V,X):
    #print (np.shape (vmat(np.append(1, X))))
    R=np.dot(W,np.vstack((np.ones((1,X.shape[1])),X)))

    #dot realiza la multiplicacion entre las matrices
    Z=np.zeros(R.shape)
    Z=sigmoid(R)
    # la funcionc append agrega un elemento a una  arreglo
    Y=np.dot(V,np.vstack((np.ones((1,X.shape[1])),Z)))

    return Y, R, Z


def entrenaMLP(W,V,X,D,alpha):

    #crea las matrices de los respectivos tamaños
    DW = np.zeros (np.shape(W))
    DV = np.zeros (np.shape(V))


    for i in range(X.shape[1]):

        #x=np.transpose(np.atleast_2d(X[i,:]))
        x=vmat(X[:,i])
        Y, R, Z, = operaMLP(W,V,x)
        e= (((D[:,i])-Y.T)).T
        DV=DV+alpha*(np.dot(e,vmat(np.append(1, Z)).T))
        #dV = dV+alpha*e*[1; Z]';
        #dW = dW+alpha*e*(V(2:end)'.*(sech(R).^2))*[1; X(:,i)]';
        DW=DW+alpha*(np.dot ((np.multiply(np.dot(((V.T[1:])),e),(sch(R)**2))),vmat(np.append(1, X[:,i])).T))
    W=W+DW
    V=V+DV
    return W ,V

def main():
    W,V=creaMLP(2, 1, 30)
    #X=np.random.rand(8,50)
    #D=np.random.rand(2,50)
    X=np.array ([[0, 1, 0, 1], [0, 0, 1, 1]])
    #X = [0 1 0 1; 0 0 1 1];
    #print (nm)
    D=np.array([[0, 1, 1, 0]])
    #print (nm1)
    #entrenaMLP(W,V,X,D, 0.01)
    #print (V.shape)
    Y,R, Z=operaMLP(W,V,X)
    e= (sum ((sum(abs (D-Y)))))
    print (e)
    dietime=1
    while dietime <=10000:
        W,V=entrenaMLP(W,V,X,D, 0.001)
        Y,R, Z=operaMLP(W,V,X)
        #print (e)
        e= (sum ((sum(abs (D-Y)))))
        dietime +=1
    #mostrar los pesos  
    print (Y)
    return 0 

main()