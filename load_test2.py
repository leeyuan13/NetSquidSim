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

atom_times = [[1e6, 1e3],]
rep_times = [[1e6, 1e3], [1e6, 1e3]]
channel_loss = 0.10

# Load from file.
DIRECTORY = 'single_run2/'

num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18]
num_repeats = 1
duration = 1000

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
# rate[0][i][k][j] = average rate with num_qubits[i] qubits, up to the j-th entanglement, for the k-th
#				  trial, for the hybrid protocol
# rate[1][i][k][j] = same as above, but for the traditional / QuTech protocol
# Note that rate will be a list of lists, not an ndarray.
# We expect one entanglement to be generated every 1/rate time steps, on average.
rates = [[], []]
for i in range(len(num_qubits)):
	# Need the +1's in the numerator because of Python zero-indexing.
	# Need the +1's in the denominator because time = 0 does not mean infinite rate; it means
	# one per time step.
	rates[0].append([[(j+1)/(data_hybrid[i][k][2][j][0]+1) for j in range(len(data_hybrid[i][k][2]))]\
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

