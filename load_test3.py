import numpy as np
import matplotlib.pyplot as plt
import pickle

from hybrid8 import run_simulation as run_hybrid
from traditional1 import run_simulation as run_trad

num_repeaters = [1, 3, 5, 7, 9, 11]
num_qubits = [2,]
num_repeats = 5
#num_qubits = [200,]
#num_repeats = 2
duration = 100
directory = 'T2_dom2/'

data = dict()
data[directory] = load_test2_data(directory, num_qubits, num_repeats)
avg_fidelities = data[directory][0][3] # (hybrid, trad)
mean_hybrid_fidelities = np.mean(avg_fidelities[0], axis = 1, keepdims = True)
mean_trad_fidelities = np.mean(avg_fidelities[1], axis = 1, keepdims = True)
ratio_fidelities = mean_hybrid_fidelities / mean_trad_fidelities
# To estimate statistics, do all pairwise ratios.
# Assume 5 repeats, and num_qubits defined above.
all_ratio_fidelities = [[avg_fidelities[0][k][i] / avg_fidelities[1][k][j] for i in range(num_repeats)\
								for j in range(num_repeats)] for k in range(len(num_qubits))]
mean_ratio_fidelities = np.mean(all_ratio_fidelities, axis = 1, keepdims = True)
stdev_ratio_fidelities = np.std(all_ratio_fidelities, axis = 1, keepdims = True)

text = ''
for i in range(len(num_qubits)):
	text += str(np.abs(mean_ratio_fidelities[i][0])) + '\t'
text = text[:-1] + '\n'
for i in range(len(num_qubits)):
	text += str(np.abs(stdev_ratio_fidelities[i][0])) + '\t'
print(text)

