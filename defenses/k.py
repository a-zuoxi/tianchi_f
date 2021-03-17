import visdom
import numpy as np

vis = visdom.Visdom(env = "sdd7")
x = [i for i in range(10)]
print(x)
update = 'append'
vis.line(x,x,update=update, win="123xx24")
vis.save(['sdd7'])
x = [i for i in range(10,20)]
y = [-i for i in range(20, 30)]
vis.line(x,y,update=update, win="123xx24")

