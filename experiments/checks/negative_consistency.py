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
    mean = [300,400]
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
