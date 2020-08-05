import numpy as np
import matplotlib.pyplot as plt
import pickle

from hybrid8 import run_simulation as run_hybrid
from traditional1 import run_simulation as run_trad
# args: num_channels (= 2*num_qubits for run_hybrid,
#					  = num_qubits for run_trad)
#		atom_times = [[T1, T2] for Alice's and Bob's electron spins]
#		rep_times = [[T1, T2] for electron spin, [T1, T2] for nuclear spin]
#		channel_loss ~ probability of losing a photon in the link from Alice/Bob to the repeater
#						(goes into amplitude damping)
# All times in nanoseconds (?).
from load_test2c import load_test2_data

#num_qubits = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
#num_qubits = [46, 48, 50, 52, 54, 56]
#num_qubits = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18]
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

