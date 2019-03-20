import numpy as np
from timeit import default_timer as timer
from numba import vectorize

# This should be a substantially high value. On my test machine, this took
# 33 seconds to run via the CPU and just over 3 seconds on the GPU.
NUM_ELEMENTS = 100000000

# This is the CPU version.
def vector_add_cpu(a, b):
    c = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    for i in range(NUM_ELEMENTS):
        c[i] = a[i] + b[i]
    return c

# This is the GPU version. Note the @vectorize decorator. This tells
# numba to turn this into a GPU vectorized function.
@vectorize(["float32(float32, float32)"], target='cuda')
def vector_add_gpu(a, b):
    return a + b;

def main():
    a_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
    b_source = np.ones(NUM_ELEMENTS, dtype=np.float32)

    # Time the CPU function
    start = timer()
    vector_add_cpu(a_source, b_source)
    vector_add_cpu_time = timer() - start

    # Time the GPU function
    start = timer()
    vector_add_gpu(a_source, b_source)
    vector_add_gpu_time = timer() - start

    # Report times
    print("CPU function took %f seconds." % vector_add_cpu_time)
    print("GPU function took %f seconds." % vector_add_gpu_time)

    return 0

if __name__ == "__main__":
    main()
