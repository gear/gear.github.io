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

### Thinking the "hard" way :wink:

## Compilation Notes

### Cuda requirements

### Linking runtime libraries

### Makefiles

## Example

### Element-wise Array Operations

### 1D Convolution
