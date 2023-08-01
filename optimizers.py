import numpy as np

class SGD:
    def __init__(self, point, dfunc, lr = 1e-4):
        self.point = point
        self.dfunc = dfunc
        self.lr = lr
        
    def step(self):
        grad = self.dfunc(self.point)
        self.point -= grad*self.lr
    
class MomentumSGD:
    def __init__(self, point, dfunc, alpha = 0.9, lr = 1e-4):
        self.point = point
        self.dfunc = dfunc
        self.lr = lr
        self.dir = 0
        self.alpha = alpha
        
    def step(self):
        grad = self.dfunc(self.point)
        self.dir = self.dir * self.alpha + (1 - self.alpha) * self.lr*grad
        self.point -= self.dir
        
class RMSprop:
    def __init__(self, point, dfunc, alpha = 0.9, eps = 1e-8, lr = 1e-3):
        self.point = point
        self.dfunc = dfunc
        self.lr = lr
        self.dir = 0
        self.alpha = alpha
        self.eps = eps
        
    def step(self):
        grad = self.dfunc(self.point)
        self.dir = self.dir * self.alpha + grad*grad * (1 - self.alpha)
        self.point -= self.lr*grad/np.sqrt(self.dir + self.eps)
        
        
class AdaGrad:
    def __init__(self, point, dfunc, eps = 1e-8, lr = 1e-2):
        self.point = point
        self.dfunc = dfunc
        self.lr = lr
        self.dir = 0
        self.eps = eps
        
    def step(self):
        grad = self.dfunc(self.point)
        self.dir += grad**2
        self.point -= self.lr*grad/np.sqrt(self.dir + self.eps)