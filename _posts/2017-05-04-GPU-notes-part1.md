---
layout: post
type: math
title: GPU Programming Notes - Part 1
subtitle: "GPU programming mental image"
---

May the 4th be with you.

It is always exciting to start programming, whether it is a new language or a
whole new scheme. This post is my mental image of (Nvidia) GPU programming; it
belongs to my series of GPU Programming Notes:
1. Part 1: Nvidia GPU programming mental image. (this post)
2. [Part 2: Kernels and Shared Memory.]({{site.baseurl}}) (coming soon)
3. [Part 3: 2d convolutional neural network.]({{site.baseurl}}) (coming soon)
4. Coming later.

## Programming Abstraction and Hardware Abstraction

I personally find it easier to have a concrete mental image when I learn
something new. For GPU (Cuda) programming, it helps a lot when I finally
see the connection between programming abstraction (grid, block, threads) and
hardware abstraction (streaming multiprocessor, streaming processor).

![GPU Software vs Hardware]({{site.baseurl}}/img/gpu-software-hardware.svg)

### Thinking the programmer's way

The first concept in GPU programming is __block__. Sometimes we might hear
parallel computing people talk about _blocking_. I think it depends on the
context to know whether _blocking_ means synchronization (as in threads
synchronization), or "trying with different block sizes to increase performance"
(as in telling the GPU to run with _this number_ of blocks).
To put it simply, __block__ can be think of as a computing unit.
Your GPU program (or function, or __kernel__) is executed by threads in a
__block__.

![GPU grids and blocks]({{site.baseurl}}/img/gpu-grid-block.svg)

The diagram above depicts the hierarchy of computation in a GPU. Although block
is an intermediate concept ("smaller" than grid, "bigger" than warp), I think
it is the most important. To the extent of my experience so far, one need
to think how his/her kernel (program) will be executed on a block first, then
the rest will come easier.

As a convention, the GPU(s) on a machine is called __device(s)__, the machine
itself (containing CPU and main memory) is called __host__. In a computation
program, the _host code_ handles data input, memory allocation, and invoke the
_device code_. The execution of _device code_ is called __kernel launch__,
where a kernel is a "small" function that can be mapped onto the GPU and
executed in parallel.

![GPU program anatomy]({{site.baseurl}}/img/gpu-prog-anatomy.svg)

A basic kernel in CUDA starts with the keyword `__global__`, meaning the
function is available for both host and device.

```cpp
/* Define a kernel */
__global__ void cudaAddVectors(float *a, float *b, float *c) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index]
}

/* Launch a kernel */
cudaAddVectors<<<grid_dim, block_dim>>>(a_mem, b_mem, c_mem);
```

In the example above, the function which will be ran on GPU is `cudaAddVectors`.
Each function (kernel) is assigned to a thread (within a block), we
have built-in variables to uniquely identify the threads, some of them are:
`threadIdx`, `blockIdx`, and `blockDim`. `threadIdx` is a unique thread
identifier within a block. `blockIdx` is a unique block identifier with in a
grid. `blockDim` is the number of threads defined in a block, this number is
defined in the second parameter of kernel launch (in `<<< >>>`). Programmer uses
these unique identifier to map the computation to the input array. In the
example above, each thread is mapped to one array element in each array input.
The figure below demonstrate the mapping:

```
            -------------------------------------------------------------
 Mem. Index | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|
            -------------------------------------------------------------
            -------------------------------------------------------------
ThreadIdx.x | 0| 1| 2| 3| 4| 0| 1| 2| 3| 4| 0| 1| 2| 3| 4| 0| 1| 2| 3| 4|
            -------------------------------------------------------------
                                                             blockDim = 5
            ---------------+--------------+--------------+---------------
 BlockIdx.x |      0       |      1       |      2       |      3       |
            ---------------+--------------+--------------+---------------
```

It is clear from the example above that `index = blockIdx.x * blockDim.x +
threadIdx.x`. Since we define `blockDim=5`, each block has 5 threads. The `.x`
is the x axis of each variable. In CUDA, every computation unit is up to
3D-index. Behind the scene, everything is 1D. However, sometimes it is more
convenient to have 2D or 3D indexing depending on the application. In the code
snippet above, a kernel call is parameterized with two integers: `grid_dim` and
`block_dim`. `grid_dim` is the number of blocks, while `block_dim` is the number
of threads per block. All these are in `.x` axis. To explicitly specify
`gridDim` and `blockDim`, we can use the provided `dim3` datatype:

```cpp
dim3 blockSize(64, 16); // C++ syntax to create block of shape (64,16,1)
dim3 gridSize = {64,64,1}; // C syntax to create grid of shape (64,64,1)
cudaAddVectors<<<gridSize, blockSize>>>(a_mem, b_mem, c_mem);
/* Kernel cudaAddVectors will be mapped to a grid containing 64 blocks
 * on x axis and y axis (64^2 blocks in total). Each of these blocks has
 * 64 threads on x axis and 16 threads on y axis.
 */
```

In summary, we have a few keywords in this section:
- `block`: The basic computing unit containing threads that execute `kernels`.
- `kernel`: The function that will be run in parallel on GPU.
- `device`: Usually refers to the GPU and code runs on GPU.
- `host`: Usually refers to the CPU and code runs on CPU.
- `thread`: The smallest computing unit. Organized as groups of 32 in a block.
- `grid`: The computing unit contains blocks.

### Thinking the "hard" way :wink:

As a tech savvy, I enjoy looking at the specification of a new graphic card:

```
Nvidia GeForce GTX 1080 Ti:
– Graphics Processing Clusters: 6
– Streaming Multiprocessors: 28
– CUDA Cores (single precision): 3584
– Texture Units: 224
– ROP Units: 88
– Base Clock: 1480 MHz
– Boost Clock: 1582 MHz
– Memory Clock: 5505 MHz
– Memory Data Rate: 11 Gbps
– L2 Cache Size: 2816K
– Total Video Memory: 11264MB GDDR5X
– Memory Interface: 352-bit
– Total Memory Bandwidth: 484 GB/s
– Texture Rate (Bilinear): 331.5 GigaTexels/sec
– Fabrication Process: 16 nm
– Transistor Count: 12 Billion
– Connectors: 3 x DisplayPort, 1 x HDMI
– Form Factor: Dual Slot
– Power Connectors: One 6-pin, One 8-pin
– Recommended Power Supply: 600 Watts
– Thermal Design Power (TDP): 250 Watts
– Thermal Threshold: 91° C
```

The specs above belongs to the new GTX 1080 Ti. It has 28 Streaming
Multiprocessors (SM). Each streaming multiprocessor has 128 CUDA cores, making
total of 3584 cores on the device. In the programming abstraction, we think in
blocks, here, we think in Streaming Multiprocessors. __Behind the scene, the
computation for each block is mapped to a SM.__ These processor can be think of
as a less complex CPU that can schedule and dispatch multiple threads at one.

About memory, GTX 1080 Ti has a whooping 11GB of global memory (they release it
right after our lab's order for Titan X 12GB :unamused:). Global memory, as it
name, can be accessed by all streaming multiprocessors (hence, all blocks when
we think in software abstraction). It is very much like RAM in normal computer.
This memory is the slowest memory in a GPU.

Faster memory compared to global memory is Cache (L1 and L2), shared memory,
local memory, and register. Register is extremely fast (few nanosecond access time). 32 or 64 32-bit registers are available to a thread. L1 and shared memory
are built with the same hardware. Their sizes are configurable. In GTX 1080 Ti,
shared memory has maximum size of 96KB. Shared memory management is one of the
optimization methods. Local memory is extra storage for registers, L2 is
a cache for global and local memory (I do not know in further detail about
these type of memory).

Knowing the hardware architecture is always helpful for programmer. More
information on GTX 1080 Ti can be found in the [white paper](http://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_1080_Whitepaper_FINAL.pdf).

## Compilation Notes

I find studying an example Makefile is the best way to learn compilation for a
new framework.

```make
# Link all runtime (rt) libraries
LD_FLAGS = -lrt
# CUDA code generation floags
GENCODE_FLAGS := -gencode arch=compute_20,code=sm20 # Compute capacity 2.0
# CUDA paths
CUDA_PATH = /usr/local/cuda
# CUDA headers
CUDA_INC_PATH := $(CUDA_PATH)/include
# CUDA tools binary (e.g. nvcc compiler)
CUDA_BIN_PATH := $(CUDA_PATH)/bin
# CUDA libraries .so files
CUDA_LIB_PATH := $(CUDA_PATH)/libraries

# C++ compiler
CC = /usr/bin/g++
# CUDA compiler
NVCC = $(CUDA_BIN_PATH)/nvcc
# For 64-bit operating system
NVCCFLAGS := -m64

# Grouping of targets
TARGET = save_the_world

all: $(TARGET)

save_the_world: save_the_world_host.cpp utils.cpp save_world.o
  $(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

save_world.o: save_world.cu
  $(NVCC) $(NVCCFLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<  

# $^ - All dependencies
# $@ - Name of the target
# $< - First dependency
```

### Cuda requirements

Installing CUDA framework is quite simple. The following code install CUDA-8.0.
Note that we need to have nvidia driver installed before installing CUDA.

```
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

## Example

There are some examples to conclude this note.

### Element-wise Array Operations

Elements.

### 1D Convolution

Sounds.
