Resource allocation
===================

This repo accompanies the paper
[_Allocation of Fungible Resources via a Fast, Scalable Price Discovery Method_](https://web.stanford.edu/~boyd/papers/resource_alloc.html).

To get started with the code, clone this repo, run

```
python setup.py install
```

in a virtual environment of your choice, and try out the notebooks, which reproduce the examples from the paper.

## Example
The `resalloc` package exports one main class representing a resource allocation problem, called `AllocationProblem`. It also exports a number of utility functions.

Here is a code example showing how to set up and solve a simple problem.

```python3
import torch
from resalloc.fungible import AllocationProblem, utilites

n_jobs, n_resources = int(1e6), 4
throughput_matrix = torch.rand((n_jobs, n_resources))
resource_limits = torch.rand(n_resoures) * n_jobs + 1e3

problem = AllocationProblem(
  throughput_matrix=throughput_matrix,
  resource_limits=resource_limits,
  utility_function=utilities.Log()
)

problem.solve(verbose=True)

# X is the optimal allocation
print(problem.X)

# prices are the optimal prices
print(problem.prices)
```

For more details about the available utilities, and how to customize the solve method with optional arguments, please consult the source code.
