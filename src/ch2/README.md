# chapter2 CUDA入门

代码架构:

```bash
ch2/
- hello.cu / hello.cpp  # cpp迁移cuda
- hello2.cu             # 核函数
- hello3.cu             # 多线程/多线程块 
- hello4.cu             # 多维 + 时间观察
- hello5.cu             # 线程顺序打乱输出
```


## 核函数和线程组织

一个核函数的典型例子:
```cpp
__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}
```

指定对应的线程块和每块线程数量
> 三括号中的第一个数字  可以看作线程块的个数,第二个数字可以看作每个线程块中的线程数。

```
<<<n, m>>>
表示n个线程块, 每个里面m个线程, 所以核函数执行n*m遍
```

实际上具体的执行顺序很迷:
```bash
root@g17:~/cuda-learning# ./build/ch2/hello3
Hello from GPU from block 1, thread 0!
Hello from GPU from block 1, thread 1!
Hello from GPU from block 1, thread 2!
Hello from GPU from block 1, thread 3!
Hello from GPU from block 3, thread 0!
Hello from GPU from block 3, thread 1!
Hello from GPU from block 3, thread 2!
Hello from GPU from block 3, thread 3!
Hello from GPU from block 2, thread 0!
Hello from GPU from block 2, thread 1!
Hello from GPU from block 2, thread 2!
Hello from GPU from block 2, thread 3!
Hello from GPU from block 0, thread 0!
Hello from GPU from block 0, thread 1!
Hello from GPU from block 0, thread 2!
Hello from GPU from block 0, thread 3!
```

貌似每一块的线程是一起的, 但是块之间的顺序不一定一起, 具体调度需要进一步研究

块内的输出为什么总是有序的, 块之间的输出为什么不会交叉, 这可能是缓冲区作为独立单位导致的

在多维中可以体现, grid和block的维度都可以设置为3维的结构体

```cpp

    dim3 grid(1, 2, 1);
    dim3 block(1, 1, 3);
    printf("Launching kernel with %d blocks and %d threads per block\n", 
        grid.x * grid.y * grid.z, block.x * block.y * block.z);

    hello_from_gpu<<<grid, block>>>();

```

坐标从内层到外层分别是x, y, z, 可以理解为三重循环从内向外的顺序, x维度变化多, z维度变化少

观察`hello4.cu`的输出

```bash
Launching kernel with 2 blocks and 3 threads per block
Hello from CPU
Hello from GPU from block (0, 1, 0) and thread (0, 0, 0)
Hello from GPU from block (0, 1, 0) and thread (0, 0, 1)
Hello from GPU from block (0, 1, 0) and thread (0, 0, 2)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 0)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 1)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 2)
Sleeping for 1000000000 seconds
Sleeping for 1000000000 seconds
Sleeping for 1000000000 seconds
Sleeping for 1000000000 seconds
Sleeping for 1000000000 seconds
Sleeping for 1000000000 seconds
Finished sleeping in block (0, 0, 0) and thread (0, 0, 0)  start clock is (243581283)
Finished sleeping in block (0, 0, 0) and thread (0, 0, 1)  start clock is (243581283)
Finished sleeping in block (0, 0, 0) and thread (0, 0, 2)  start clock is (243581283)
Finished sleeping in block (0, 1, 0) and thread (0, 0, 0)  start clock is (243574817)
Finished sleeping in block (0, 1, 0) and thread (0, 0, 1)  start clock is (243574817)
Finished sleeping in block (0, 1, 0) and thread (0, 0, 2)  start clock is (243574817)
Hello from CPU
```

会发现这个块内的线程的创建都是同步的, 块之间的时间差非常大, 目前的代码不会导致块内顺序倒转

而块之间的顺序可以通过while循环空转来实现改变.


从开普勒架构开始,最大允许的线程块大小是$1024$,而最大允许的网格大小是$2^{31}-1$(针对这里的一维网格来说;多维网格能够定义更多的线程块)。

CUDA 中对能够定义的网格大小和线程块大小做了限制。对任何从开普勒到图灵架构的 GPU 来说,网格大小在 x、y 和 z 这 3 个方向的最大允许值分别为 $2^{31}-1$、$65535$ 和 $65535$;  线程块大小在 x、y 和 z 这 3 个方向的最大允许值分别为 $1024$、$1024$ 和 $64$。另外还要求线程块总的大小,即 blockDim.x、blockDim.y 和 blockDim.z 的乘积不能大于$1024$。也就 是说,不管如何定义,一个线程块最多只能有$1024$个线程。


## 疑问: 能否实现块内线程顺序输出翻转?


```bash
Launching kernel with 2 blocks and 3 threads per block
Hello from CPU
Hello from GPU from block (0, 1, 0) and thread (0, 0, 0)
Hello from GPU from block (0, 1, 0) and thread (0, 0, 1)
Hello from GPU from block (0, 1, 0) and thread (0, 0, 2)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 0)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 1)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 2)
blockIdx: (1), threadIdx:(0) Sleeping for 9000000000 seconds
blockIdx: (1), threadIdx:(1) Sleeping for 6000000000 seconds
blockIdx: (1), threadIdx:(2) Sleeping for 3000000000 seconds
blockIdx: (0), threadIdx:(0) Sleeping for 10000000000 seconds
blockIdx: (0), threadIdx:(1) Sleeping for 7000000000 seconds
blockIdx: (0), threadIdx:(2) Sleeping for 4000000000 seconds
Finished sleeping in block (0, 1, 0) and thread (0, 0, 0)  start clock is (232812645)
Finished sleeping in block (0, 1, 0) and thread (0, 0, 1)  start clock is (232812645)
Finished sleeping in block (0, 1, 0) and thread (0, 0, 2)  start clock is (232812645)
Finished sleeping in block (0, 0, 0) and thread (0, 0, 0)  start clock is (232818522)
Finished sleeping in block (0, 0, 0) and thread (0, 0, 1)  start clock is (232818522)
Finished sleeping in block (0, 0, 0) and thread (0, 0, 2)  start clock is (232818522)
Hello from CPU
```

发现简单的时间差反转似乎不行.

GPT给出的理由如下:
```markdown
你的观察是正确的：仅靠 sleep（忙等待）并不能保证块内线程的输出顺序被翻转。原因如下：

- CUDA 的线程是并发执行的，printf 的输出顺序无法严格控制。
- sleep（忙等待）只是让线程大致延后执行，但不能阻止其他线程先输出。
- printf 在 GPU 上是异步的，输出缓冲区的刷新顺序也不保证。
```

所以我们通过显式设置输出顺序即可控制线程输出:
```cpp
__global__ void hello_from_gpu() {
    // 线程按 threadIdx.x 从大到小输出
    for (int i = blockDim.x - 1; i >= 0; --i) {
        if (threadIdx.x == i) {
            printf("Hello from GPU from block (%d, %d, %d) and thread (%d, %d, %d)\n",
                   blockIdx.x, blockIdx.y, blockIdx.z,
                   threadIdx.x, threadIdx.y, threadIdx.z);
        }
        __syncthreads();
    }
}
```

输出结果如下:
```bash
Launching kernel with 2 blocks and 3 threads per block
Hello from CPU
Hello from GPU from block (0, 1, 0) and thread (2, 0, 0)
Hello from GPU from block (0, 0, 0) and thread (2, 0, 0)
Hello from GPU from block (0, 1, 0) and thread (1, 0, 0)
Hello from GPU from block (0, 0, 0) and thread (1, 0, 0)
Hello from GPU from block (0, 1, 0) and thread (0, 0, 0)
Hello from GPU from block (0, 0, 0) and thread (0, 0, 0)
Hello from CPU
```