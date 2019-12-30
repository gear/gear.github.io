---
layout: post
title: "Szemeredi's Regularity Lemma"
description: "Terence Tao revisited the Regularity Lemma."
tags: [egt, math]
---

This post summarizes the paper "Szemer\'edi's Regularity Lemma 
Revisited" by Terence Tao.  The [paper](https://arxiv.org/abs/math/0504472) is
published on arXiv.

---

## Draft

Szemeredi Regularity Lemma states that for dense graphs, we can specify an error rate $0 < \epsilon < 1$  such that the graph can be partitioned into $O(1)$ parts. Among these partitions, the edges behaves almost randomly. Such partition
is called $\epsilon$-regular. 

In this paper, the author views Szemeredi Regularity Lemma from the information
theoric perspective. Such point of view not only clarifies the Regularity Lemma,
but also allows applications of other techniques (information theory, ergodic
theory) to proving and improving the regularity lemma. 

Regularity Lemma is a building stone to the Szemeredi Theorem. The theorem
states that in a [positive density](https://en.wikipedia.org/wiki/Natural_density)
subset of natural numbers, there exists an arbitarily large arithmetic progression.
This results relates strongly to the 
[ergodic theory](https://en.wikipedia.org/wiki/Ergodic_theory).

Instead of studying dense graph, in this paper, the author uses graphs and 
bi-paritite graphs as an analogue to random variables. Suppose there are two 
random variables $x_1$ and $x_2$ mapping from the vertices space 
($V_1$ and $V_2$) to some probability measurement. 

There is a recent paper from Cambridge (probably sumitted to ICML 2019) that
connect ergodic theory with machine learning. It was published on 
[arXiv](https://arxiv.org/pdf/1811.07192.pdf) with the title "The Theory and
Algorithm of Ergodic Inference".

(Dec 26 -- paused)

### 1. Introduction

## Main ideas and contributions

1. Starting from a skewed stacked RNN architecture, the authors proposed **a novel
RNN where each hidden unit is parameterized by a high rank tensor (≥2)**.

    ![sRNN and tRNN]({{site.baseurl}}/img/trnn.png)

    The image above compares a three hidden layers (depth L = 3) skewed recurrent neural network with the proposed model.
    The tRNN model here has only one hidden layer, but its hidden units are parameterized by a P-by-M matrix rather than
    a single vector. Furthermore, the interaction between hidden units are captured by the convolution operation (defined
    later) to take advantages of the tensor representation. According to the authors, the first dimension of the hidden units is "locally connected" in order to share parameters, while the second dimension is "fully connected" for global interaction. This idea will be clearer when we discuss the convolution. In general, the construction of a hidden unit output is computed as following ($$\circledast$$ is the convolution operator):
   
   $$
    H^\text{cat}_{t-1, p} = \left\{ 
    \begin{array}{l}
    x_t W^x + b^x, \text{ if } p = 1 \\ 
    h_{t-1, p-1}, \text{ else } %_
    \end{array}
    \right. \ \ \ \
    \begin{array}{l}
    A_t = H^\text{cat}_{t-1} \circledast \{W^h, b^h\} \\
    H_t = \phi(A_t) \\
    y_t = \varphi(h_{t+L-1, p} W^y + b^y)
    \end{array}
   $$

    Note that the actual output at $$y_t$$ is computed using the last tensor (in this case it is a vector) of 
    the hidden tensor $$H_{t+L-1}$$. As in the figure above, $$y_t$$ is computed from $$h^3_{t+2}$$ - the last 
    layer of the hidden unit $$H_{t+2}$$. Also, the implicit depth P = L = 3 leads us to the same computation.  

2. The paper introduces **cross-layer convolution and memory cell convolution (for the LSTM extension)**. The explanation
to the convolution is presented in the second section. Note that in this post, I moved the time notation to the top of
each symbol when it's convenient so that the subscript contains only indexing variables (e.g. $$A_{t, p, m^o}$$ becomes $$A^t_{p, m^o}$$).

3. The 2D-tRNN model is further **extended to LSTM and higher order tensors (3D)**.

    $$
    \begin{array}{l}
    [A^g_t, A^i_t, A^f_t, A^o_t] = H^\text{cat}_{t-1} \circledast \{W^h, b^h\} \\
    [G_t, I_t, F_t, O_t] = [\phi{A^g_t}, \sigma{A^i_t}, \sigma{A^f_t}, \sigma{A^o_t}] \\
    \text{Memory cell: } C_t = G_t \odot I_t + C_{t-1} \odot F_t \in R^{P \times M} \\
    H_t = \phi(C_t) \odot O_t 
    \end{array}
    $$

    This extension to LSTM is pretty straight-forward as each gate is computed using the convolution operator
    $$\circledast$$ in lieu of the standard matrix multiplication. For the higher order tensors extension, the concatenated
    tensor is constructed by appending the projection (multiplied with weights, added with bias) of $$x_t$$ to one 
    _corner_ of the hidden unit tensor. This can be understood as going down the tensor dimension-wise, when you reach
    a 2D matrix with row size of M, append $$x_t W^x + b^x$$ to it. In the same way, $$y_t$$ is generated with the
    opposite corner. 

## Dissecting convolution

It took me sometime to fully grasp the author's idea :dizzy_face:. Despite that "sometime", my understanding can be
captured in a single image (and now I feel silly). In the case of 2D hidden units, the operation and the dimensionality 
of each tensor is:

$$
A_t = H^\text{cat}_{t-1} \circledast \{W^h, b^h\}, \text{ where: } \ \ \  
\begin{array}{l}
A_t \in R^{P \times M^o} \\
H^\text{cat}_{t-1} \in R^{(P+1) \times M} \\
W^h \in R^{K \times M^i \times M^o} \\
b^h \in R^{M^o}
\end{array}
$$

In here, $$A_t$$ is the activation tensor. $$H^\text{cat}_{t-1}%_$$ represents the concatenation of a hidden unit's 
rank-2 tensor output (matrix) and the input vector from the layer above it (in this case, it is the input vector 
$$x_t$$). From the skewed sRNN figure above, the concatenation is pretty clear: $$h_t$$ is pointed to by $$x_t$$ and
$$h_{t-1}$$, hence the tensor used in computing $$A_t$$ and $$h_t$$ will be the concatenation between $$x_t$$ and
$$h_{t-1}$$. The same principle applies for the tRNN. The convolution kernel consists of weight $$W^h$$ and bias $$b^h$$.
$$W^h$$ is a rank-3 tensor of K filters, each filters has $$M^i$$ input channels and $$M^o$$ output channels. In this
paper, the authors noted that they let $$M = M^i = M^o$$ for simplicity. The detail of the convolution is given in
the supplementary document:

$$
A^t_{p, m^o} = \sum^K_{k=1} \left( \sum^{M^i}_{m^i=1} H^{cat; t-1}_{p - \frac{K-1}{2}+k, m^i} \cdot W^k_{k,m^i,m^o}  \right) + b^h_{m^o}
$$

It might sound obvious, but it is easier for me to think of the indices as "selector". For example, $$A^t_{p, m^o}$$
is the element $$\square$$ of tensor $$A^t$$ that is selected by $$(p, m^o)$$. Usually, the "selector" is lowercase
and the total number of element in a dimension is uppercase. In the figure below, I circled in red the "selectors".

![sRNN and tRNN]({{site.baseurl}}/img/rnn_conv.png)

It might helps with a kind of story. The matrix $$A^t \in R^{P \times M^o}$$ is what we need to compute for the next time 
step. We set out to compute each element of this matrix. $$A^t_{p, m^o}$$ is given by the convolution operator defined
in the formula above. In the figure, the gray small box represents $$A^t_{p, m^o}$$. The two selector $$p$$ and
$$m^o$$ control the center of the convolution in $$H^{\text{cat}; t-1}$$ and which output channel to pick in $$W^h$$ 
respectively. For example, if $$K=5$$ and where want to computer $$A^t$$ at $$(p=3, m^o=3)$$, then for each slice of
$$W^h$$, the 5 columns associated with $$m^o=3$$ will be picked out. Next, on $$H^{\text{cat}; t-1}$$, there are also 5 
rows selected. Since $$p=3$$, the index of these row vectors are $$\{2,3,4,5,6\}$$ (centered at $$p+1=4$$). These 5 pairs
of vectors are indexed by $$k = \{1,2,3,4,5\}$$. Sum of the dot products of the pairs is $$A^t_{p, m^o}$$.
Note that the authors use zero-padding to keep the shape of output same as the input. In the case of memory cell 
convolution, the values used in padding are the border values to avoid interference with the memory values. 

## Evaluations

The authors uses three main tasks to demonstrate the effectiveness of their proposed model:

1. **Algorithmic tasks** consist of learning to sum two 15-digit numbers and learning to repeat the input sequence. 

    ```
     Sum two 15-digit integers    ||   Repeat the input
     Input : --2545-9542-------   ||   Input : -hiworld----------
     Output: -----------12087--   ||   Output: --------hiworld--- 
    ```
    The advantage of the proposed model (3D-tLSTM with Channel Normalization) is that it requires much less training
    samples to reach more than 99 percent accuracy compared with Stacked LSTM and Grid LSTM. However, there are no
    mention to the training time or other modification of tLSTM. Furthermore, the best performing models have 7 and 10
    layers depth. There are not much explanation for these hyper-parameters. The 7-layer requires less training samples
    than the 10-layer in addition task, but the 10-layer in turn requires much less samples in memorization. I expected
    the 7-layer model requires less samples in both tasks.

2. **MNIST classification** tasks consist of normal MNIST and randomized pixel sequence called pMNIST. In these tasks, 
the pixels of an hand-written digit image are fed into the neural nets as a sequence. In this task, I do not see any
advantage of using tLSMT compared with state-of-the-art methods such as DilatedGRU. 

3. **Wikipedia language modeling** task deal with next-character prediction on the Hutter Prize Wikipedia dataset. Similar
to the aforementioned MNIST tasks, there are no clear win for tLSTM compared with Large FS-LSTM-4 method (in term of
both BPC and number of parameters).

## Why tensors?

It is true that we can design a "vector" recurrent neural net that works in the same way as the proposed tRNN here. After
all, the variables is stored in arrays on our computer. However, tensors represent a concrete way to think about a "block
of numbers", and more importantly, we can define and argue about convolution easily with it. Let's take image data as 
an example. The image below shows the popular 2D convolution operation. Tensor representation gives us a concrete way
to think about color channels and its corresponding channels in each convolution kernel. Furthermore, notice that we
"scan" each convolution channel in the intensity dimension (on the matrices) but not the color dimension. As such, the
weights in the intensity dimension are shared. Similarly, in recurrent neural networks, the tensor hidden unit in some
sense enables a more abstract representation and allow weight sharing.

![sRNN and tRNN]({{site.baseurl}}/img/2d_conv.png)

## Conclusion 

This paper proposes a nice way to increase the structural depth of a recurrent neural network model. Instead of 
parameterizing hidden units with vectors, tRNN modeled hidden units as tensors for more efficient weight sharing
and implicit depth. Such approach greatly reduces the number of parameters in a recurrent neural network while 
maintaining the depth. tRNN model is also extended to use LSTM module and higher order (3D) tensor for better performance
and data abstraction. The effectiveness of tLSTM is demonstrated on three different tasks in which tLSTM achieve 
state-of-the-art performance with some improvement on number of parameters (minor) and required number of training
samples. On the other hand, while the proposed model might improve running time and number of parameters, there was no
discussion on the training time and training complexity of tLSTM. It would be interesting to implement the 2D and 3D models
to understand the benefit of tLSTM better.

In this post, I wrote about my understanding and commented on the paper "Wider and Deeper, Cheaper and Faster:
Tensorized LSTMs for Sequence Learning" by Zhen He, Shaobing Gao, Liang Xiao, Daxue Liu, Hangen He, and David Barber.
I left out some details such as Channel Normalization or the constraints on $$\{L, P, K\}$$. These minor optimization
tricks can be found on the paper. After this post, I plan to implement this technique to see if it is suitable for my
current work. In particular, I would like to see if the training cost can potentially be reduced for a stacked LSTM
approach to model product names.
