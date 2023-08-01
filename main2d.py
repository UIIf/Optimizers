import matplotlib.pyplot as plt
import numpy as np
from tasks2d import task1
from optimizers import SGD, MomentumSGD, RMSprop, AdaGrad
from utils import OptimHandler, PlotDrawer2d, OptimDrawer, get_points2d

steps = 300
dt = 0.02
ruiner_count = 50
np.random.seed(42)

x_range = [-0.05, 1.05]

func, dfunc, ddfunc = task1()
x, y = get_points2d(x_range[0], x_range[1], 400, func)
plot_drawer = PlotDrawer2d(x, y, c = dfunc(x), s = 0.4)

optims = [SGD, MomentumSGD, RMSprop, AdaGrad]
rand_points = np.random.rand(ruiner_count)
for opt in optims:
    oh = OptimHandler(rand_points, opt, dfunc=dfunc)
    od = OptimDrawer(oh, func, label = opt.__name__, alpha = 0.3)
    
    plot_drawer.draw_list += [od.draw]
    plot_drawer.after_draw_list += [oh.step]    

for t in range(steps):
    plot_drawer.draw_frame(dt)
plt.show()

