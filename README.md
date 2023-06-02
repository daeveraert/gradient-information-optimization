# GIO: Gradient Information Optimization
<p align="center">
  <img alt="" src="https://github.com/daeveraert/gradient-information-optimization/blob/main/images/process.gif">
</p>

GIO is a library that implements Gradient Information Optimization (GIO) at scale. GIO is a data selection technique that can
be used to select a subset of training data that gives similar or superior performance to a model trained on full data.

**Features**:
- GIO with quantization using K-means.
- Sentence embedding script to generate embeddings from data to use in GIO

## Installation

Installable via pip:
```bash
pip install grad-info-opt
``` 
Or install directly form the repository:

```bash
git clone git@github.com:daeveraert/gradient-information-optimization.git
cd gradient-information-optimization
pip install -e .
```

Direct installation will require you to install additional dependencies listed below. We welcome contributions to GIO.

## Requirements
- `numpy>=1.21.6`
- `jax>=0.3.25`
- `pyspark>=2.4.8`
- `sentence_transformers>=2.2.2`
- `jaxlib>=0.4.7`
- `pandas>=1.0.5`



## Quick Start
**Note:** GIO uses a Spark context, or if it can't find one, it will create a local one. You may encounter a Spark error before the algorithm runs complaining it cannot find a free port. In this case, executing ```export SPARK_LOCAL_IP="127.0.0.1"``` should resolve the issue.

Here is a simple 2D demonstration of how to use GIO with visualization:
```python
from GIO import GIOKL
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Create some data
def getX():
    mean = [3,4]
    cov = [[0.5,0],[0,0.5]]
    np.random.seed(1)
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    return jnp.array([[x[i],y[i]] for i in range(len(x))])

def getXTest():
    mean = [3,4]
    cov = [[0.5,0],[0,0.5]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    return jnp.array([[x[i],y[i]] for i in range(len(x))])

X = getX()
X_test = getXTest()

# Initialize class
gio_kl = GIOKL.GIOKL(uniform_low=0, uniform_high=8, uniform_start_size=100, dim=2)

# Perform the Algorithm
W, kl_divs, _ = gio_kl.fit(X_test, X, normalize=False)
W = W[100:] # Remove the uniform start

# Plot results
plt.plot(kl_divs)
plt.title("KL Divergence vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.show()
plt.clf()
plt.scatter([each[0] for each in W], [each[1] for each in W], label='Selected Data')
plt.scatter([each[0] for each in X], [each[1] for each in X], label='Target Data')
plt.title("Target Data and Selected Data")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
```
<p align="center">
  <img alt="" src="https://github.com/daeveraert/gradient-information-optimization/blob/main/images/readme_ex1.png" width="49%">
  <img alt="" src="https://github.com/daeveraert/gradient-information-optimization/blob/main/images/readme_ex2.png" width="49%">
</p>

Here is a more complex example for scale applications, reading and using a CSV that stores embeddings and data, using quantization-explosion, and Spark:
```python
from GIO import GIOKL
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyspark.sql.functions as F

# Initialize class
gio_kl = GIOKL.GIOKL(uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=768)

# Read data
train_df, target_df = gio_kl.read_data_from_csv(PATH_TO_TRAIN, PATH_TO_TARGET)

# Quantize data
model_train, model_X, transformed_train, transformed_X = gio_kl.quantize(train_df, target_df)

X = jnp.array(model_X.clusterCenters())
train = jnp.array(model_train.clusterCenters())
centroids_df = gio_kl.spark.createDataFrame(data=[(i, each.tolist()) for i, each in enumerate(model_train.clusterCenters())], schema=["id", "centroid"])

# Perform the Algorithm
W, kl_divs, _ = gio_kl.fit(train, X, max_iter=300, stopping_criterion='sequential_increase_tolerance', v_init='jump')
W = W[20:] # Remove the uniform start

# Explode back to original data and write resulting data
full_selections_df = gio_kl.explode(W, transformed_train, centroids_df)
full_selections_df.select(F.col("_c0"), F.col("_c1")).write.option("delimiter", "\t").csv(OUTPUT_PATH)


# Plot results
plt.plot(kl_divs)
plt.title("KL Divergence vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.show()
```
**Note:** For quantization, Spark requires a large rpc message size. It is recommended to place ```gio_kl.spark.conf.set("spark.rpc.message.maxSize", "500")```  (or any large number) in the code before calling quantize, if the defaults haven't already been increased.

## Available Options
`GIOKL.fit` takes the following arguments:
- `train`: training data as a jnp array (jnp is almost identical to numpy) [M, D] shape
- `X`: target data as a jnp array [N, D] shape
- `D`: initial data as a jnp array, default None. Use None to initialize from 0 (uniform) or a subset of training data
- `k`: kth nearest neighbor to use in the KL divergence estimation, default 5
- `max_iter`: maximum iterations for the algorithm. One iteration adds one point (cluster)
- `stop_criterion`: a string for the stopping criterion, one of the following: 'increase', 'max_resets', 'min_difference', 'sequential_increase_tolerance', 'min_kl', 'data_size'. Default is 'increase'
    - `min_difference`: the minimum difference between prior and current KL divergence for 'min_difference' stop criterion only. Default is 0
    - `resets_allowed`: whether if KL divergence increases, resetting G to the full train is allowed (allows the algorithm to pick duplicates). Must be set to true if the stop criterion is 'max_resets'. Default is False
    - `max_resets`: the number of resets allowed for the 'max_resets' stop criterion only (a reset resets G to the full train set and allows the algorithm to pick duplicates). Default is 2
    - `max_data_size`: the maximum size of data to be selected for the 'data_size' stop criterion only, as a percentage (of total data) between 0 and 1. Default is 1
    - `min_kl`: the minimum kl divergence for the 'min_kl' stop criterion only. Default is 0
    - `max_sequential_increases`: the maximum number of sequential KL divergence increases for the 'sequential_increase_tolerance' stop criterion only. Default is 3
- `random_init_pct`: the percent of training data to initialize the algorithm from. Default is 0
- `random_restart_prob`: probability at any given iteration to extend the gradient descent iterations by 3x, to find potentially better extrema. Higher values come at the cost of efficiency. Default is 0
- `scale_factor`: factor to scale the gradient by, or 'auto'. Default is 'auto', which is recommended
- `v_init`: how to initialize v in gradients descent, one of the following: 'mean', 'prev_opt', 'jump'. Default is 'mean'
- `grad_desc_iter`: the number of iterations to use in gradient descent. Default is 50
- `discard_nearest_for_xy`: discard nearest in the xy calculation of KL divergence, for use when X and the train set are the same, comes at the cost of efficiency. Default is False
- `lr`: Learning rate for gradient descent. Default is 0.01

## Citing GIO
If you use GIO in a publication or blog, please cite this software.
```
@software{gradient-information-optimization,
  author = {Dante Everaert},
  title = {GIO: Gradient Information Optimization for Training Dataset Selection},
  url = {https://github.com/daeveraert/gradient-information-optimization},
  version = {0.1.0},
  year = {2023},
  note = {Apache 2.0 License}
}
```
