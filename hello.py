import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate random dataset
np.random.seed(42)
X = 2 * np.random.rand(10, 1)
# y = 4 + 3 * X + np.random.randn(10, 1)
y = [10,20,30,0,-10,-20,-30,30,40,50]
# X = [i for i in range(1, len(y))]

# Initialize parameters
theta = np.zeros((2, 1))  # Zero initialization of theta
learning_rate = 0.1
n_iterations = 1000
m = len(X)

# Add x0 = 1 to each instance (for the bias term)
X_b = np.c_[np.ones((m, 1)), X]

# Function to compute the cost (Mean Squared Error)
def compute_cost(theta, X_b, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X_b.dot(theta) - y) ** 2)

# Gradient Descent
def gradient_descent(X_b, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = np.zeros(n_iterations)
    theta_history = np.zeros((n_iterations, 2))

    for i in range(n_iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(theta, X_b, y)
        theta_history[i] = theta.ravel()
        
    return theta, cost_history, theta_history

# Run gradient descent
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, learning_rate, n_iterations)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
scat = ax.scatter(X, y)
line, = ax.plot(X, X_b.dot(theta), color='red')
title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
ax.set_xlabel('X')
ax.set_ylabel('y')

def update(frame):
    if frame % 10 == 0:
        line.set_ydata(X_b.dot(theta_history[frame]))
        title.set_text(f'Iteration {frame}\nLoss = {cost_history[frame]:.4f}\nTheta0 = {theta_history[frame][0]:.4f}, Theta1 = {theta_history[frame][1]:.4f}')
    return line, title

ani = animation.FuncAnimation(fig, update, frames=n_iterations, blit=True, interval=100)

# Show the plot
plt.show()

# Plot cost history
plt.figure(figsize=(8, 6))
plt.plot(range(n_iterations), cost_history)
