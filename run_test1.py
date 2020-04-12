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

atom_times = [[1e9, 1e6],]
rep_times = [[1e9, 1e6], [1e9, 1e6]]
channel_loss = 0.10

num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,] # even numbers

# Note that the entanglement signals are sent every 1 ns.

duration = 1000

#data_hybrid = []
#data_trad = []
#for m in num_qubits:
#	print('hybrid', m)
#	result = run_hybrid(int(m/2), atom_times, rep_times, channel_loss, duration).data
#	with open('run_test1_data_hybrid_'+str(m)+'.pickle', 'wb') as fn: pickle.dump(result, fn)
#	data_hybrid.append(result)
#	print('trad', m)
#	result = run_trad(m, atom_times, rep_times, channel_loss, duration).data
#	with open('run_test1_data_trad_'+str(m)+'.pickle', 'wb') as fn: pickle.dump(result, fn)
#	data_trad.append(result)
#
## Save data!
#with open('run_test1_data_hybrid.pickle', 'wb') as fn: pickle.dump(data_hybrid, fn)
#with open('run_test1_data_trad.pickle', 'wb') as fn: pickle.dump(data_trad, fn)

# data_hybrid[0] = time of the i-th entanglement with Alice
# data_hybrid[1] = time of the i-th entanglement with Bob
# data_hybrid[2] = (time of the i-th entanglement, 
#					time at which the final stage of the i-th entanglement started,
#					number of attempts made with the electron spin when the nuclear spin was in
#					place)
# data_hybrid[3] = density matrix of entangled state

def fidelity(density_matrix):
	# Computes the fidelity wrt the desired state, psiP (01+10).
	# Extract the 0, 0 element because the output will be a 1x1 matrix.
	return (np.dot(np.array([[0, 1, 1, 0]]), \
				np.dot(density_matrix, np.array([[0, 1, 1, 0]]).T))/2)[0,0]

# Step 1: compute average rates of entanglement.
# rate[0][i][j] = average rate with num_qubits[i] qubits, up to the j-th entanglement, 
#				  for the hybrid protocol
# rate[1][i][j] = same as above, but for the traditional / QuTech protocol
# Note that rate will be a list of lists, not an ndarray.
# We expect one entanglement to be generated every 1/rate time steps, on average.
rates = [[], []]
for i in range(len(num_qubits)):
	# Need the +1's in the numerator because of Python zero-indexing.
	# Need the +1's in the denominator because time = 0 does not mean infinite rate; it means
	# one per time step.
	rates[0].append([(j+1)/(data_hybrid[i][2][j][0]+1) for j in range(len(data_hybrid[i][2]))])
	rates[1].append([(j+1)/(data_trad[i][2][j][0]+1) for j in range(len(data_trad[i][2]))])

# Step 2a: compute average wait times for repeater qubits.
wait_times = [[], []]
for i in range(len(num_qubits)):
	next_times = []
	for j in range(min(len(data_hybrid[i][0]), len(data_hybrid[i][1]))):
		next_times.append(np.abs(data_hybrid[i][0][j] - data_hybrid[i][1][j]))
	wait_times[0].append(next_times)
	
	next_trad_times = []
	for j in range(min(len(data_trad[i][0]), len(data_trad[i][1]))):
		next_trad_times.append(np.abs(data_trad[i][0][j] - data_trad[i][1][j]))
	wait_times[1].append(next_trad_times)

# Step 2b: count the number of electron attempts made.
electron_attempts = [[], []]
for i in range(len(num_qubits)):
	electron_attempts[0].append([x[2] for x in data_hybrid[i][2]])
	electron_attempts[1].append([x[2] for x in data_trad[i][2]])

# Step 3: compute fidelities of the resulting states.
fidelities = [[], []]
for i in range(len(num_qubits)):
	fidelities[0].append([fidelity(x) for x in data_hybrid[i][3]])
	fidelities[1].append([fidelity(x) for x in data_trad[i][3]])

# Compute averages.
# The average rate of entanglement is the last element of the list, by definition.
avg_rates = [[x[-1] for x in rates[0]], [x[-1] for x in rates[1]]]
# The average wait time is the mean of all wait times.
avg_wait_times = [[np.mean(x) for x in wait_times[0]], [np.mean(x) for x in wait_times[1]]]
# The average number of electron attempts is the mean, as above.
avg_electron_attempts = [[np.mean(x) for x in electron_attempts[0]], \
						 [np.mean(x) for x in electron_attempts[1]]]
# The average fidelity is taken to be the arithmetic mean. (Should it be the harmonic mean??)
avg_fidelities = [[np.mean(x) for x in fidelities[0]], [np.mean(x) for x in fidelities[1]]]

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
	plt.figure()
	plt.plot(num_qubits, avg_rates[0], 'b.--', label = 'hybrid')
	plt.plot(num_qubits, avg_rates[1], 'r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel(r'average rate / clock cycles$^{-1}$')
	plt.legend()

	plt.figure()
	plt.plot(num_qubits, avg_wait_times[0], 'b.--', label = 'hybrid')
	plt.plot(num_qubits, avg_wait_times[1], 'r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel('average wait time for a match / clock cycles')
	plt.legend()

	plt.figure()
	plt.plot(num_qubits, avg_electron_attempts[0], 'b.--', label = 'hybrid')
	plt.plot(num_qubits, avg_electron_attempts[1], 'r.--', label = 'traditional')
	plt.xlabel('number of qubits in repeater')
	plt.ylabel('average number of electron attempts')
	plt.legend()

