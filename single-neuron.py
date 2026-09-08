import numpy as np

X=np.array([[1,0.2],[0.9,0.3]])
y_true=np.array([0,1]) # ground truth for verififcation

lr=0.5 # learning rate 
print(X)

w=np.array([1,2]) # arbitrary weights
b=0.5 # arbitrary bias

n=len(y_true)
z=np.dot(X,w)+b # linear function
A=1/(1+np.exp(-z)) # activation function (here sigmoid)

erreur=A-y_true 

# gradient error
dw=np.dot(X.T,erreur)/n
db=np.sum(erreur)/n

# update the weights and the bias
w=w-lr*dw
b=b-lr*db
print(b)

# forward pass again with new weights/bias
z=np.dot(X,w)+b
y_pred=1/(1+np.exp(-z)) 

print(y_pred)
