# They must accept NumPy arrays for image_size and batch (broadcasting), 
# so that plotting a surface over the (S,B) plane is a single call

import numpy as np

def flops(image_size: np.ndarray, batch: np.ndarray) -> float:
    flops_value = 17712 * image_size ** 2 * batch + 313344 * batch
    return flops_value

def memory(image_size: np.ndarray, batch: np.ndarray) -> float:
    memory_value = 26 * 4 * batch * image_size ** 2 + 1040324 * 4
    return memory_value

def latency(image_size: np.ndarray, batch: np.ndarray, theta: np.ndarray) -> float:
    flops_value = flops(image_size, batch)
    bytes_value = 4 * (91 * batch * image_size ** 2 + 1040324)
    theta0, theta1, theta2 = theta
    latency_value = theta0 + np.maximum(flops_value * theta1, bytes_value * theta2)
    return latency_value

def energy(image_size: np.ndarray, batch: np.ndarray, theta_latency: np.ndarray, theta_energy: np.ndarray):
    theta3, theta4, theta5= theta_energy
    lat = latency(image_size, batch, theta_latency)
    flops_value = flops(image_size, batch)
    mem = memory(image_size, batch)
    return theta3 * lat + theta4 * flops_value + theta5 * mem