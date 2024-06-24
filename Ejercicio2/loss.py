import numpy as np

class MSE:
    def apply(self,y,a):
        temp = a - y
        return (1. / (2. * y.shape[1])) * np.sum(temp * temp)

    def derivative(self,y,a):
        return (1. / y.shape[1]) * np.sum(a - y, axis=1).reshape((y.shape[0],1))


class CrossEntropy:
    def apply(self,y,a):
        a = np.clip(a,1e-10,1-1e-10)    #preventing division by zero
        return (-1. / y.shape[1]) * np.sum(y*np.log(a) +(np.ones(y.shape)-y)*np.log(np.ones(a.shape)-a))

    def derivative(self,y,a):
        a = np.clip(a,1e-10,1-1e-10)    #preventing division by zero
        return (1. / y.shape[1]) * (np.divide(-y,a)+np.divide(np.ones(y.shape)-y,np.ones(a.shape)-a))