import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_test2_data(DIRECTORY, num_qubits, num_repeats):
	# Note that the entanglement signals are sent every 1 ns.
	
	# data_hybrid[i][k][0] = time of the i-th entanglement with Alice
	# data_hybrid[i][k][1] = time of the i-th entanglement with Bob
	# data_hybrid[i][k][2] = (time of the i-th entanglement, 
	#					time at which the final stage of the i-th entanglement started,
	#					number of attempts made with the electron spin when the nuclear spin was in
	#					place)
	# data_hybrid[i][k][3] = density matrix of entangled state
	
	data_hybrid = []
	data_trad = []
	for m in num_qubits:
		data_hybrid.append([])
		data_trad.append([])
		for k in range(num_repeats):
			with open(DIRECTORY+'run_test2_data_hybrid_'+str(m)+'_trial_'+str(k)+'.pickle', 'rb') as fn: 
				result = pickle.load(fn)
			data_hybrid[-1].append(result)
			with open(DIRECTORY+'run_test2_data_trad_'+str(m)+'_trial_'+str(k)+'.pickle', 'rb') as fn: 
				result = pickle.load(fn)
			data_trad[-1].append(result)
	
	def fidelity(density_matrix):
		# Computes the fidelity wrt the desired state, psiP (01+10).
		# Extract the 0, 0 element because the output will be a 1x1 matrix.
		return (np.dot(np.array([[0, 1, 1, 0]]), \
					np.dot(density_matrix, np.array([[0, 1, 1, 0]]).T))/2)[0,0]
	
	# Step 1: compute average rates of entanglement.
	# rate[0][i][k][j] = average rate with num_qubits[i] qubits, up to the j-th entanglement, 
	#					for the k-th trial, for the hybrid protocol
	# rate[1][i][k][j] = same as above, but for the traditional / QuTech protocol
	# Note that rate will be a list of lists, not an ndarray.
	# We expect one entanglement to be generated every 1/rate time steps, on average.
	rates = [[], []]
	for i in range(len(num_qubits)):
		# Need the +1's in the numerator because of Python zero-indexing.
		# Need the +1's in the denominator because time = 0 does not mean infinite rate; it means
		# one per time step.
		rates[0].append([[(j+1)/(data_hybrid[i][k][2][j][0]+1) \
							for j in range(len(data_hybrid[i][k][2]))]\
							for k in range(num_repeats)])
		rates[1].append([[(j+1)/(data_trad[i][k][2][j][0]+1) for j in range(len(data_trad[i][k][2]))]\
						for k in range(num_repeats)])
	
	# Step 2a: compute average wait times for repeater qubits.
	wait_times = [[], []]
	for i in range(len(num_qubits)):
		wait_times[0].append([])
		wait_times[1].append([])
		for k in range(num_repeats):
			next_times = []
			for j in range(min(len(data_hybrid[i][k][0]), len(data_hybrid[i][k][1]))):
				next_times.append(np.abs(data_hybrid[i][k][0][j] - data_hybrid[i][k][1][j]))
			wait_times[0][-1].append(next_times)
			
			next_trad_times = []
			for j in range(min(len(data_trad[i][k][0]), len(data_trad[i][k][1]))):
				next_trad_times.append(np.abs(data_trad[i][k][0][j] - data_trad[i][k][1][j]))
			wait_times[1][-1].append(next_trad_times)
	
	# Step 2b: count the number of electron attempts made.
	electron_attempts = [[], []]
	for i in range(len(num_qubits)):
		electron_attempts[0].append([[x[2] for x in data_hybrid[i][k][2]] for k in range(num_repeats)])
		electron_attempts[1].append([[x[2] for x in data_trad[i][k][2]] for k in range(num_repeats)])
	
	# Step 3: compute fidelities of the resulting states.
	fidelities = [[], []]
	for i in range(len(num_qubits)):
		fidelities[0].append([[fidelity(x) for x in data_hybrid[i][k][3]] for k in range(num_repeats)])
		fidelities[1].append([[fidelity(x) for x in data_trad[i][k][3]] for k in range(num_repeats)])
	
	# Compute averages.
	# The average rate of entanglement is the last element of the list, by definition.
	avg_rates = [[[x[-1] for x in rates[0][j]] for j in range(len(num_qubits))], 
				 [[x[-1] for x in rates[1][j]] for j in range(len(num_qubits))]]
	# The average wait time is the mean of all wait times.
	avg_wait_times = [[[np.mean(x) for x in wait_times[0][j]] for j in range(len(num_qubits))], \
					  [[np.mean(x) for x in wait_times[1][j]] for j in range(len(num_qubits))]]
	# The average number of electron attempts is the mean, as above.
	avg_electron_attempts = [[[np.mean(x) for x in electron_attempts[0][j]]\
																	for j in range(len(num_qubits))], \
							 [[np.mean(x) for x in electron_attempts[1][j]]\
							 										for j in range(len(num_qubits))]]
	# The average fidelity is taken to be the arithmetic mean. (Should it be the harmonic mean??)
	avg_fidelities = [[[np.mean(x) for x in fidelities[0][j]] for j in range(len(num_qubits))], \
					  [[np.mean(x) for x in fidelities[1][j]] for j in range(len(num_qubits))]]

	return (avg_rates, avg_wait_times, avg_electron_attempts, avg_fidelities), \
			(rates, wait_times, electron_attempts, fidelities)

num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
num_repeats = 5
duration = 10000
directory = 'tests2/T2_domA3/'

data = dict()
data[directory] = load_test2_data(directory, num_qubits, num_repeats)

avg_fidelities = data[directory][0][3] # (hybrid, trad)
mean_hybrid_infidelities = 1-np.mean(avg_fidelities[0], axis = 1, keepdims = True)
mean_trad_infidelities = 1-np.mean(avg_fidelities[1], axis = 1, keepdims = True)
ratio_infidelities = mean_hybrid_infidelities / mean_trad_infidelities
# To estimate statistics, do all pairwise ratios.
# Assume 5 repeats, and num_qubits defined above.
all_ratio_infidelities = [[(1-avg_fidelities[0][k][i]) / (1-avg_fidelities[1][k][j]) for i in range(num_repeats)\
								for j in range(num_repeats)] for k in range(len(num_qubits))]
mean_ratio_infidelities = np.mean(all_ratio_infidelities, axis = 1, keepdims = True)
stdev_ratio_infidelities = np.std(all_ratio_infidelities, axis = 1, keepdims = True)

# Part 0: num_qubits.
text0 = ''
for num in num_qubits:
	text0 += str(num) + '\t'
#text0 = text0[:-1] + '\n'
print(text0)

# Part 1: estimate fidelity improvement.
text = ''
for i in range(len(num_qubits)):
	text += str(np.abs(mean_ratio_infidelities[i][0])) + '\t'
text = text[:-1] + '\n'
for i in range(len(num_qubits)):
	text += str(np.abs(stdev_ratio_infidelities[i][0])) + '\t'
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
	text3 += str(np.abs(mean_hybrid_infidelities[i][0])) + '\t'
text3 = text3[:-1] + '\n'
for i in range(len(num_qubits)):
	text3 += str(np.abs(mean_trad_infidelities[i][0])) + '\t'
print(text3)


