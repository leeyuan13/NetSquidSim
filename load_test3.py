import numpy as np
import matplotlib.pyplot as plt
import pickle

num_repeaters = [1, 3, 5, 7, 9, 11]
num_qubits = 2,
num_repeats = 1
duration = 2e8
directory = 'run_test6_1km/'

data = dict()
for nr in num_repeaters:
	for k in range(num_repeats):
		with open(PREFIX+'run_test6_data_hybrid_'+str(nr)+'_trial_'+str(k)+'.pickle', 'wb') as fn:
			data[(nr, k)] = pickle.load(fn)

rates = []
fidelities = []
for nr in num_repeaters:
	times, dms = data[(nr, 0)]
	rates.append(len(times)/times[-1])
	indiv_fidelities = [0.5*(dm[1, 1]+dm[1, 2]+dm[2, 1]+dm[2, 2]) for dm in dms]
	fidelities.append(np.mean(indiv_fidelities))

plt.figure()
plt.plot(num_repeaters, rates, '.:')
plt.xlabel('Number of repeaters')
plt.ylabel(r'Rate (s$^{-1}$)')
plt.title('2 qubits per router, 1km between routers')

plt.figure()
plt.plot(num_repeaters, fidelities, '.:')
plt.xlabel('Number of repeaters')
plt.ylabel('Mean fidelity')
plt.title('2 qubits per router, 1km between routers')
