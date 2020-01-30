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

num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18]

params = {'T2_dom1/': {'T1': 1e12, 'T2': 1e-3, 'num_qubits': num_qubits[:], 'channel_loss': 0.10,\
						'num_repeats': 5, 'duration': 1000},
		  'T2_dom2/': {'T1': 1e12, 'T2': 1e0, 'num_qubits': num_qubits[:], 'channel_loss': 0.10,\
		  				'num_repeats': 5, 'duration': 1000},
		  'T2_dom3/': {'T1': 1e12, 'T2': 1e3, 'num_qubits': num_qubits[:], 'channel_loss': 0.10,\
		  				'num_repeats': 5, 'duration': 1000}}

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

def plot_times(input_data):
	time1 = input_data[0]
	time2 = input_data[1]
	num_trials = min(len(time1), len(time2))
	REPEATER_CONTROL_RATE = 1
	wait_time = [abs(time1[i] - time2[i])/REPEATER_CONTROL_RATE for i in range(num_trials)]

	plt.figure()
	plt.hist(wait_time, bins = list(range(int(max(wait_time))+2)))
	plt.xlabel('wait time for a successful match / clock cycles')
	plt.ylabel('number of occurrences')
	#plt.title('Wait time between matches across channels')

	successful_entanglement = [x[0] for x in input_data[2]]
	plt.figure()
	plt.plot(range(1, len(time1)+1), [x+1 for x in time1], 'b:')
	plt.plot(range(1, len(time2)+1), [x+1 for x in time2], 'r:')
	plt.plot(range(1, len(successful_entanglement)+1), \
			[x+1 for x in successful_entanglement], 'k-')
	plt.xlabel('number of successes')
	plt.ylabel('time of occurrence')

	plt.figure()
	plt.plot(range(1, len(successful_entanglement)+1), \
			[(i+1)/(successful_entanglement[i]+1) for i in range(len(successful_entanglement))], 'k.')
	plt.xlabel('number of successes')
	plt.ylabel('rate of successes')

	electron_attempts = [x[2] for x in input_data[2]]
	plt.figure()
	plt.hist(electron_attempts, bins = list(range(int(max(electron_attempts)+1))))
	plt.xlabel('number of electron attempts with nuclear spin occupied')
	plt.ylabel('number of occurrences')

def plot_data():
	# Plot mean and standard deviation!
	plt.figure()
	plt.errorbar(num_qubits, np.mean(avg_rates[0], axis = 1), yerr=np.std(avg_rates[0], axis = 1), \
					fmt='b.--', label = 'hybrid')
	plt.errorbar(num_qubits, np.mean(avg_rates[1], axis = 1), yerr=np.std(avg_rates[1], axis = 1),\
					fmt='r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel(r'rate of entanglement (clock cycles$^{-1}$)')
	plt.legend()

	plt.figure()
	plt.errorbar(num_qubits, np.mean(avg_wait_times[0], axis = 1), \
					yerr=np.std(avg_wait_times[0], axis = 1),\
					fmt='b.--', label = 'hybrid')
	plt.errorbar(num_qubits, np.mean(avg_wait_times[1], axis = 1), \
					yerr=np.std(avg_wait_times[1], axis = 1),\
					fmt='r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel('wait time for a match (clock cycles)')
	plt.legend()

	plt.figure()
	plt.errorbar(num_qubits, np.mean(avg_electron_attempts[0], axis=1), \
					yerr=np.std(avg_electron_attempts[0], axis=1), fmt='b.--', label = 'hybrid')
	plt.errorbar(num_qubits, np.mean(avg_electron_attempts[1], axis=1),\
					yerr = np.std(avg_electron_attempts[1], axis=1), fmt='r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel('number of electron attempts')
	plt.legend()

if False:
	data = dict()
	for directory in params:
		print(directory)
		num_qubits = params[directory]['num_qubits']
		num_repeats = params[directory]['num_repeats']
		data[directory] = load_test2_data(directory, num_qubits, num_repeats)
	
if False:
	directory_list = ['T2_dom'+str(i)+'/' for i in range(1, 4)]
	color_list = [(0.2, 0, 0.2), (0.5, 0, 0.5), (0.8, 0, 0.8)]
	label_list = [r'$T_2 = 10^{-3}$', r'$T_2 = 10^{0}$', r'$T_2 = 10^{3}$']
	# Save to file in MATLAB-compatible format.
	text = ''
	# Number of qubits.
	for num in num_qubits:
		text += str(num) + '\t'
	text = text[:-1] + '\n'
	# Mean ratios, then stdevs.
	# Plot fidelities, if needed.
	plt.figure()
	for i in range(3):
		directory = directory_list[i]
		avg_fidelities = data[directory][0][3] # (hybrid, trad)
		mean_hybrid_fidelities = np.mean(avg_fidelities[0], axis = 1, keepdims = True)
		mean_trad_fidelities = np.mean(avg_fidelities[1], axis = 1, keepdims = True)
		ratio_fidelities = mean_hybrid_fidelities / mean_trad_fidelities
		# To estimate statistics, do all pairwise ratios.
		# Assume 5 repeats, and num_qubits defined above.
		all_ratio_fidelities = [[avg_fidelities[0][k][i] / avg_fidelities[1][k][j] for i in range(5)\
										for j in range(5)] for k in range(len(num_qubits))]
		mean_ratio_fidelities = np.mean(all_ratio_fidelities, axis = 1, keepdims = True)
		stdev_ratio_fidelities = np.std(all_ratio_fidelities, axis = 1, keepdims = True)
	
		print('----')
		print(ratio_fidelities)
		print(mean_ratio_fidelities)

		for rat in mean_ratio_fidelities:
			text += str(np.abs(rat[0])) + '\t'
		text = text[:-1] + '\n'
		for rat in stdev_ratio_fidelities:
			text += str(np.abs(rat[0])) + '\t'
		text = text[:-1] + '\n'

		# Again, assume same num_qubits in all parameter settings.
		plt.errorbar(num_qubits, mean_ratio_fidelities, yerr=stdev_ratio_fidelities, fmt='.--',\
					 label = label_list[i], color = color_list[i])
	plt.xlabel('number of qubits in repeater')
	plt.ylabel('hybrid / traditional fidelity')
	plt.legend()
	plt.show()

if True:
	# Extract rates.
	text2 = ''
	for num in num_qubits:
		text2 += str(num) + '\t'
	text2 = text2[:-1] + '\n'
	
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
	text2 = text2[:-1] + '\n'
