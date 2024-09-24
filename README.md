# PicoGrad: The "Tiny" Autograd Engine

## Because size doesn't always matter in ML

PicoGrad is a tiny autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG, with a focus on speed and support for distributed tensors. It's a supercharged version of [micrograd](https://github.com/karpathy/micrograd) with a few extra bells and whistles.

We called it "pico" for the same reason you might call your gaming PC a "little setup" – pure, delightful understatement.

## Features

- Implements a general-purpose Tensor class that supports distributed computing
- Supports dynamic computational graph construction
- Provides automatic differentiation (autograd) capabilities
- Includes basic neural network building blocks
- Offers graph optimization for improved performance

## Extending to Neural Networks

PicoGrad can be used to build, train neural networks and visualize the computational graph.

![trained model](./graphs/trained_model.png)


## Distributed Tensor Support

PicoGrad goes beyond micrograd by supporting distributed tensors, allowing for efficient computation on large datasets:

```python
from picograd.engine import DistributedTensor
import numpy as np

data = DistributedTensor(np.random.rand(1000000, 1000))
print(data)
```

## Graph Optimization

PicoGrad includes graph optimization techniques to improve computational efficiency:


![initial graph](./graphs/initial_graph.png)

![optimized graph](./graphs/optimized_graph.png)