import matplotlib.pyplot as plt
import numpy as np
import time
from generate_system import GenerateBody
from simulate_orbits import evolve

start = time.time()
bodyList = [1]
for i in range(len(bodyList)): bodyList[i] = GenerateBody()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.ion()
plt.show()


def update_plot():
    ax1.clear()
    ax1.plot(bodyList[0].xpoints,bodyList[0].ypoints)
    ax1.scatter(bodyList[0].xpoints[-1],bodyList[0].ypoints[-1], color='red',edgecolor=None)
    ax1.scatter(0, 0, color='yellow')
    plt.draw()
    plt.pause(0.0001)

t = 0
dt = 60*60*24  # 1 day
while t < (dt * 3 * 365):
    t +=dt
    evolve(bodyList,time,dt)
    if t % (dt*5) == 0: update_plot() # update plot every 5 days

end = time.time()
print(end-start)
print(t/dt , ' days')