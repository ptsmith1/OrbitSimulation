import numpy as np
from timeit import default_timer as timer
from numba import guvectorize, cuda
from generate_system import *
import math

def output_gpu_data():
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))

@guvectorize(['void(float64[:,:], float64[:], float64[:], int64, float64[:])'],
             '(b,d),(d),(b),()->(d)', nopython=True, target='parallel')
def calc_acc_parallel(pos_array, my_pos_array, mass_array, minimum_interaction_distance, acc_out):
    acc_out[0] = 0
    acc_out[1] = 0
    acc_out[2] = 0
    # if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0 and cuda.blockIdx.x == 0:
    #     print("My blockID, threadID(x,y,z):", cuda.blockIdx.x, cuda.threadIdx.x,cuda.threadIdx.y,cuda.threadIdx.z)
    #     print("Pos array: [", my_pos_array[0], my_pos_array[1], my_pos_array[2], "]")
    #     print("Acc array: [", acc_out[0], acc_out[1], acc_out[2], "]")
    #     print("Mass array", mass_array[0], mass_array[1], mass_array[2])
    b, d = pos_array.shape
    G = 6.67408e-11
    for m_planet in range(b):
        xsep = my_pos_array[0] - pos_array[m_planet, 0]
        ysep = my_pos_array[1] - pos_array[m_planet, 1]
        zsep = my_pos_array[2] - pos_array[m_planet, 2]
        norm_separation = math.sqrt(xsep * xsep + ysep * ysep + zsep * zsep)
        if norm_separation >= minimum_interaction_distance:
            norm_separation_cubed = norm_separation * norm_separation * norm_separation
            ax = -((G * mass_array[m_planet] * xsep) / norm_separation_cubed)
            ay = -((G * mass_array[m_planet] * ysep) / norm_separation_cubed)
            az = -((G * mass_array[m_planet] * zsep) / norm_separation_cubed)
            acc_out[0] = acc_out[0] + ax
            acc_out[1] = acc_out[1] + ay
            acc_out[2] = acc_out[2] + az
            # if cuda.threadIdx.x == 1 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0 and cuda.blockIdx.x == 0:
            #     print("My blockID, threadID(x,y,z):", cuda.blockIdx.x, cuda.threadIdx.x, cuda.threadIdx.y,
            #           cuda.threadIdx.z)
            #     print("ax",  acc_out[0])

sun_mass = 1.989e30
planet_count = 1000000
dt = np.int32(3600*24)
run_time = 10
planets = []
planets.append(GenerateStar(0, sun_mass))
for i in range(planet_count - 1):
    planets.append(GenerateRandomBody(i + 1, sun_mass))

mass_array = np.asarray([planets[i].mass for i in range(planet_count)], dtype=np.float64)
pos_array = np.stack([planets[i].position for i in range(planet_count)], axis=0)
vel_array = np.stack([planets[i].velocity for i in range(planet_count)], axis=0)
acc_array = np.zeros_like(pos_array, dtype=np.float64)
output_gpu_data()


for i in range(1):
    acc_array = np.zeros_like(pos_array, dtype=np.float64)
    pos_array = np.add(pos_array, vel_array * dt)
    start = timer()
    calc_acc_parallel(pos_array, pos_array, mass_array, np.int64(1e10), acc_array)
    print("Time taken", timer() - start)
    vel_array = np.add(vel_array, acc_array * dt)
    print(pos_array[1], vel_array[1], acc_array[1], mass_array[1])

