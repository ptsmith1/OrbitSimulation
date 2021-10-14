import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from generate_system import *
from simulate_orbits import SystemEvolve

start = time.time_ns()
"""Simulation parameters"""
axes_limit = 2000e9
planet_count = 1000  # planets to create, max 9 if using our solar system
plot_interval = 25  # in days
dt = 3600 * 24  # in seconds
run_time = 100 # in years

frames = ((run_time*365)//plot_interval)+1
planets = []
lines = []
plots = []
labels = []
times = []
times.append(time.time_ns())
time_steps = []
xpoints = []
colours = ['gray', 'orange', 'blue', 'chocolate','brown','hotpink','black','green','violet']
system = SystemEvolve()
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
ax1.set_xlim(-axes_limit, axes_limit)
ax1.set_ylim(-axes_limit, axes_limit)
time_plot = ax2.plot(xpoints, time_steps)
timestamp = ax1.text(.03, .94, 'Day: ', color='b', transform=ax1.transAxes, fontsize='x-large')

for i in range(planet_count):
    """
    Generates the planets and their associated plotting objects, planets are either generated with random velocity/positions or are based off our solar system
    """
    colour = random.randint(0,8)
    #planets.append(GenerateStockBody(i+1))
    planets.append(GenerateRandomBody(i+1))
    plots.append(ax1.scatter([], [], color=colours[colour], s=10))
    line, = ax1.plot([], [], color=colours[colour], linewidth=1)
    label = ax1.annotate(planets[i].id, xy=(planets[i].position[:2]))
    labels.append(label)
    lines.append(line)


sun = ax1.scatter([0], [0], color='yellow', s=100)


def animate(i,plots,labels,lines,planets,start,times,xpoints,time_steps,time_plot):
    """
    This is called by the funcAnimation method from matplotlib, which calls it repeatedly to make a live animation.
    Frames is the number of frames in the animation but anim is actually called at least once more at the start of the
    animation. Therefore it is tricky to keep track of the simulation time and so i am keeping time with each planet
    rather than based on the number of frames that have been made.
    """
    delete_plots(planets,plots,lines,labels)  # deletes old plots
    plot_time(i, times, time_steps, xpoints, time_plot)  # creates the data for, and updates the time plot

    print("dt(frame) = ",time_steps[i]," ms")
    year = planets[-1].time/(3600*24*365)

    if year > run_time:
        end = time.time_ns()
        print("total time: ", (end - start)/1e9)
        plt.close()

    for n, planet in enumerate(planets):
        for count in range(plot_interval):
            SystemEvolve.evolve(system, planet)
        orbital_radius = planet.orbital_radius/149597870700
        plots[n].set_offsets(planet.position[:2])
        lines[n].set_xdata(planet.xpoints)
        lines[n].set_ydata(planet.ypoints)
        labels[n].set_x(planet.position[0])
        labels[n].set_y(planet.position[1])
        labels[n].set_text(str(round(orbital_radius,3)) + 'AU')

    timestamp.set_text('Year: ' + str(round(year,4)))
    return lines + plots + [timestamp] + labels + time_plot

def plot_time(i, times, time_steps, xpoints, time_plot):
    times.append(time.time_ns())
    time_steps.append((times[i+1]-times[i])/1e6)
    xpoints.append(len(times)-1)
    time_plot[0].set_xdata(xpoints)
    time_plot[0].set_ydata(time_steps)
    ax2.set_xlim(0, len(xpoints))
    ax2.set_ylim(0, max(time_steps)+10)

def delete_plots(planets,plots,lines,labels):
    to_delete = []
    for count, planet in enumerate(planets):
        if planet.orbital_radius/149597870700 > 50:
            to_delete.append(count)

    for count, val in enumerate(to_delete):
        planets.pop(val-count)
        plots.pop(val-count)
        lines.pop(val-count)
        labels.pop(val-count)

def normal_sim():  # use if you dont want the ~60% increased run time from the animation
    for i in range(3650):
        for planet in range(planet_count):
            SystemEvolve.evolve(system, planets[planet])
    end = time.time_ns()
    print(end-start)

ani = animation.FuncAnimation(fig, animate, repeat=False, frames=frames, fargs=(plots,labels,lines,planets,start,times,xpoints,time_steps,time_plot),blit=True, interval=0,)
#ani.save('C:/Users/Philip/Pictures/sim/animation'+str(time.time())+'.gif', writer='imagemagick', fps=60)
#normal_sim()
plt.show()
