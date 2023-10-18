import numpy as np
import matplotlib.pyplot as plt

# input the phase of the two vectors to be tested
z_1_phase = 1 / 4 * np.pi
z_2_phase = 1.2 * np.pi

# calculates the complex valued vectors
z_1 = np.exp(1j * z_1_phase)
z_2 = np.exp(1j * z_2_phase)

# calculate dot_product and normal product in C
dot_prod = z_1 * np.conjugate(z_2)
prod = z_1 * z_2


# Creating plot
figure, axes = plt.subplots()
plt.arrow(0, 0, np.real(z_1), np.imag(z_1), color='r', length_includes_head=True, width=0.01, head_width=0.05)
plt.arrow(0, 0, np.real(z_2), np.imag(z_2), color='g', length_includes_head=True, width=0.01, head_width=0.05)
plt.arrow(0, 0, np.real(dot_prod), np.imag(dot_prod), color='b', length_includes_head=True, width=0.01, head_width=0.05)
plt.arrow(0, 0, np.real(prod), np.imag(prod), color='m', length_includes_head=True, width=0.01, head_width=0.05)
circle = plt.Circle([0, 0], 1, color='r', fill=False)
axes.set_aspect(1)
axes.add_artist(circle)
plt.title('Visualization of Dot-Product vs normal product')
# x-lim and y-lim
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.grid()
plt.show()
