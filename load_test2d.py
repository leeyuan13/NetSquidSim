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
# args: directory
#       num_qubits
#       num_repeats
# returns: (avg_rates, avg_wait_times, avg_electron_attempts, avg_fidelities), \
#			(rates, wait_times, electron_attempts, fidelities)
# notes:
#       rates[0][i][k][j] = average rate with num_qubits[i] qubits, up to the j-th entanglement, 
#					        for the k-th trial, for the hybrid protocol
#       rates[1][i][k][j] = same as above, but for the traditional / QuTech protocol
#       Note that rates will be a list of lists, not an ndarray.

#       wait_times[0][i][k][j], wait_times[1][i][k][j] = waiting time for the j-th entanglement,
#                                                        for the k-th trial with num_qubits[i]
#                                                        qubits, for the appropriate protocol
#       electron_attempts[0][i][k][j], electron_attempts[1][i][k][j] = same as above, but for 
#                                                                      the number of electron attempts
#                                                                      needed
#       fidelities[0][i][k][j], fidelities[1][i][k][j] = same as above, but for Bell pair
#                                                        fidelities

#       Averages do not have the last index (by definition).

num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_repeats = 5
duration = 10000
directory = 'T2_domA1/'

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

# Part 0: num_qubits.
text0 = ''
for num in num_qubits:
	text0 += str(num) + '\t'
#text0 = text0[:-1] + '\n'
print(text0)

# Part 1: estimate fidelity improvement.
text = ''
for i in range(len(num_qubits)):
	text += str(np.abs(mean_ratio_fidelities[i][0])) + '\t'
text = text[:-1] + '\n'
for i in range(len(num_qubits)):
	text += str(np.abs(stdev_ratio_fidelities[i][0])) + '\t'
print(text)

# Part 2: extract rates.
text2 = ''
hybrid_rates = [[data[directory][0][0][0][k][II] for II in range(5)] 
						for k in range(len(num_qubits))]
trad_rates = [[data[directory][0][0][1][k][II] for II in range(5)]
						for k in range(len(num_qubits))]

avg_hybrid_rates = np.mean(hybrid_rates, axis = 1)
avg_trad_rates = np.mean(trad_rates, axis = 1)
stdev_hybrid_rates = np.std(hybrid_rates, axis = 1)
stdev_trad_rates = np.std(trad_rates, axis = 1)

for rat in avg_hybrid_rates:
	text2 += str(rat) + '\t'
text2 = text2[:-1] + '\n'
for rat in stdev_hybrid_rates:
	text2 += str(rat) + '\t'
text2 = text2[:-1] + '\n'
for rat in avg_trad_rates:
	text2 += str(rat) + '\t'
text2 = text2[:-1] + '\n'
for rat in stdev_trad_rates:
	text2 += str(rat) + '\t'
#text2 = text2[:-1] + '\n'
print(text2)

# Part 3: extract mean fidelities.
text3 = ''
for i in range(len(num_qubits)):
	text3 += str(np.abs(mean_hybrid_fidelities[i][0])) + '\t'
text3 = text3[:-1] + '\n'
for i in range(len(num_qubits)):
	text3 += str(np.abs(mean_trad_fidelities[i][0])) + '\t'
print(text3)


