# Importing all the necessary libraries and modules

import numpy as np
import math
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
from jax import jit
import pennylane as qml
import optax
from scipy.integrate import trapezoid
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

n_qubits = 10

#Activation functions

# ReLU activation function
def relu(x_array):
    x = x_array
    j = jnp.mean(x)
    y = jnp.full(x.shape[0],jnp.nan,dtype=jnp.float64)
    y = jnp.where(x <= j, 0, x - j)     # ReLU(x) = max(0,x-5)  
    y = y + 2
    area = trapezoid(y,x)
    normalized_y = y/area
    return normalized_y


#Sigmoid activation function
def modified_sigmoid(x):
    """Compute the modified sigmoid function that is zero at x=5."""
    return 1 / (1 + jnp.exp(-(x))) 

def sigmoid(x_array):
    x = x_array
    j = jnp.mean(x)    
    y = jnp.full(x.shape[0],jnp.nan,dtype=jnp.float64)
    y = modified_sigmoid(x-j)
    y = y + 5
    area = trapezoid(y,x)
    normalized_y = y/area
    return normalized_y


# Elu activation function
def modified_elu(x):
    return 1*(jnp.exp(x)-1)


def elu(x_array):
    x = x_array
    j = jnp.mean(x)
    y = jnp.full(x.shape[0],jnp.nan,dtype=jnp.float64)
    y = jnp.where(x <= j, modified_elu(x-j), x-j)     # ELU(x) = max(0,x-5)
    y = y + 5
    area = trapezoid(y,x)
    normalized_y = y/area
    return normalized_y


#Tanh activation function
def tanh(x_array):
    x = x_array
    j = jnp.mean(x_array)
    y = jnp.full(len(x),jnp.nan,dtype=jnp.float64)
    y = jnp.tanh(x-j)
    y = y + 2
    area = trapezoid(y,x)
    normalized_y = y/area
    return normalized_y

def gaussian(x_array):
    # Parameters for the Gaussian distribution
    mu = 2**(n_qubits-1)  # mean
    sigma = 2**(n_qubits-1)/4  # standard deviation

    # Gaussian distribution formula
    gaussian_distribution = (1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_array - mu) / sigma)**2))

    area = trapezoid(gaussian_distribution,x_array)
    gaussian_distribution /= area
    return gaussian_distribution