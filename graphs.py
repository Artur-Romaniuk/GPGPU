import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot Addition
vectorAddition = pd.read_csv('build/vectorAddition.csv', sep=';')
vector_sizes = vectorAddition.vs.unique()
wg32_y = vectorAddition[vectorAddition.wg==32].timing
wg64_y = vectorAddition[vectorAddition.wg==64].timing
wg128_y = vectorAddition[vectorAddition.wg==128].timing
wg256_y = vectorAddition[vectorAddition.wg==256].timing

fix, ax = plt.subplots()
ax.plot(vector_sizes, wg32_y, marker = "o", label = "work-group size 32")
ax.plot(vector_sizes, wg64_y, marker = "o", label = "work-group size 64")
ax.plot(vector_sizes, wg128_y, marker = "o", label = "work-group size 128")
ax.plot(vector_sizes, wg256_y, marker = "o", label = "work-group size 256")

ax.legend()
plt.xlabel('Vector size [elems]')
plt.ylabel('Time [ms]')
plt.show()

# Plot rotation naive
vectorRotateNaive = pd.read_csv('build/vectorRotateNaive.csv', sep=';')
matrix_sizes = vectorRotateNaive.vs.unique()
wg32_16_y = vectorRotateNaive[vectorRotateNaive.wg==32*16].timing

fix, ax = plt.subplots()
ax.plot(matrix_sizes, wg32_16_y, marker = "o", label = "work-group size 32*16")

ax.legend()
plt.xlabel('Vector size [elems]')
plt.ylabel('Time [ms]')
plt.show()

# Plot rotation optimal
vectorRotateOpt = pd.read_csv('build/vectorRotateOpt.csv', sep=';')
matrix_sizes = vectorRotateOpt.vs.unique()
wg32_16_y = vectorRotateOpt[vectorRotateOpt.wg==32*16].timing

fix, ax = plt.subplots()
ax.plot(matrix_sizes, wg32_16_y, marker = "o", label = "work-group size 32*16")

ax.legend()
plt.xlabel('Vector size [elems]')
plt.ylabel('Time [ms]')
plt.show()
