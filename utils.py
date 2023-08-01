from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np

class OptimHandler:
    def __init__(self, points:Iterable[float], optimInst, **kwargs):
        self.optimizers = [optimInst(x, **kwargs) for x in points]

    def get_x(self):
        return np.array([x.point for x in self.optimizers])
    
    def step(self):
        [x.step() for x in self.optimizers]

class PlotDrawer2d:

    def __init__(self, x:Iterable[float], y:Iterable[float], **kwargs):
        self.x = x
        self.y = y
        self.scatter_kwargs = kwargs
        self.draw_list = []
        self.after_draw_list = []
        
        self.xlim = [min(x), max(x)]
        self.ylim = [min(y), max(y)]
        delta_y = self.ylim[1] - self.ylim[0]
        self.ylim[0] -= delta_y*0.05
        self.ylim[1] += delta_y*0.05
    
    def draw(self):
        plt.scatter(self.x, self.y, **self.scatter_kwargs)
        [x() for x in self.draw_list]
        [x() for x in self.after_draw_list]
        
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.legend()
        
    def draw_frame(self, dt:float):
        plt.clf()
        self.draw()
        plt.pause(dt)
        
class OptimDrawer:
    def __init__(self, oh:OptimHandler, func, label, **scatter_kwargs):
        self.oh = oh
        self.func = func
        self.scatter_kwargs = scatter_kwargs
        self.label = label
    
    def draw(self):
        x = self.oh.get_x()
        y = self.func(x)
        
        plt.scatter(x, y, label = self.label + ": "+ str(y.mean()), **self.scatter_kwargs)

def get_points2d(x_min, x_max, x_count, func):
    x = np.linspace(x_min, x_max, x_count)
    y = func(x)
    return x, y