import numpy as np
import matplotlib.pyplot as plt
import pickle

# On the Trashcan,
# cd ~/NetsquidPackage/Sim
# Remember to run the NetSquid virtual environment for Python.
# e.g. source ~/netsquid/bin/activate

# Define IDENTIFIER, channel_length, num_repeats, duration in console.
#IDENTIFIER = '1km_0rev'
PREFIX = 'NetSquidData3/'+IDENTIFIER+'/'

num_qubits = [1000,]
#num_qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] # even numbers
#channel_length = 1 # in kilometers
#num_repeats = 1
#duration = 2e8

# We choose duration = 2e8 for 1km, 1e9 for 10km, 1e9 (1rev to 5rev) or 5e9 (6rev to 10rev) for 15km, 5e9 for 20km, 1e10 for 25km.
# Also:
#	num_qubits = 100, 30, 1000 
#					 -> channel_length = 25, 15, 10
#						num_repeats = 5
#						duration = 1e8
# 	num_qubits = 1000, channel_length = 1


print(IDENTIFIER, channel_length, num_repeats, duration)

from hybrid11 import run_simulation as run_hybrid
from traditional4 import run_simulation as run_trad
### ARGUMENTS ###
## Repeater node parameters:
# num_channels = degree of parallelism (m/2 for hybrid, m for trad)
# atom_times = [[T1, T2],] for Alice's and Bob's electron spins
# rep_times = [[T1, T2] for electron spin, [T1, T2] for nuclear spin]
# channel_loss = probability of losing a photon in the link between a client node and a detector station
#				 (goes into amplitude damping)
# duration = length of simulation in nanoseconds
# repeater_channel_loss = probability of losing a photon between a repeater qubit register
#							and a detector station in the repeater node
# noise_on_nuclear_params = [a = 1/4000, b = 1/5000] parameters for the depolarizing and dephasing
#								noise experienced by the nuclear spin when the electron spin sends
#								a photon
## Temporal parameters (in nanoseconds):
# link_delay = time of flight from a node to a detector station
# 				(Barrett-Kok requires (at least) 4 time-of-flights)
# link_time = network clock cycle (distant Barrett-Kok attempts are prompted once every 
# 									 link_time, so link_time >= 4 link_delay)
# local_delay = time of flight from a node to a local detector station
# local_time = clock cycle for local (within-repeater operations)
# time_bin = time bin resolution for beamsplitters (i.e. we count photons in a time interval of 
#				width time_bin after accounting for time of flight to the detector station)
#			 time_bin can also be the temporal resolution of the detector (e.g. dead time after
#			 	a detection event)
## Detector parameters:
# detector_pdark = dark count probability in an interval time_bin
# detector_eff = detector efficiency
## Minor parameters:
# gate_fidelity = fidelity of a single 2-qubit gate (note that SWAP is two 2-qubit gates)
# meas_fidelity = fidelity of measuring the electron spin
# prep_fidelity = fidelity of electron spin-photon entanglement

### FIXED PARAMETERS ###
# Coherence times for electron spin: T1 = 3.6e3 s, T2 = 1.58 s
#	https://www.nature.com/articles/s41467-018-04916-z.pdf
# Coherence times for nuclear spin: T1 = ? (86400 s), T2 = 63 s 
#	https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031045
atom_times = [[3.6e12, 1.58e9],]
rep_times = [[3.6e12, 1.58e9], [86.4e12, 63e9]]

# Contributions to channel efficiency:
# 	collection efficiency from NV = 0.83 * 0.66
# 	off-fibre coupling = 0.92
# 	frequency conversion = 0.33
# 	propagation loss
#		for fibre, 0.18 dB / km
# 		for repeater, 0.0159 dB / MZI, number of MZIs = log(num_qubits) + 1
# Common values for (1 - channel_loss):
# 	0 km --> 0.166
#	1 km --> 0.160
#	10 km --> 0.110
# 	20 km --> 0.0726
#	30 km --> 0.0480
# 	40 km --> 0.0317
# 	50 km --> 0.0209
# 	100 km --> 0.00264
# Common values for (1 - repeater_channel_loss): 
# 	(note that no frequency conversion is needed; assume no off-fibre coupling since all operations
#	 are on-chip)
# 	num_qubits = 2 --> 0.544
# 	num_qubits = 20 --> 0.511
fixed_channel_efficiency = 0.83 * 0.66 * 0.92 * 0.33
fixed_repeater_channel_efficiency = 0.83 * 0.66
channel_loss_per_km = 0.18 # in dB
repeater_channel_loss_per_MZI = 0.0159 # in dB

# Noise on nuclear spin from interaction with electron spin.
noise_on_nuclear_params = [1./4000, 1./5000]

# Link delays in nanoseconds.
# Assume fiber index of 1.4.
link_delay_per_km = 1e9 * 1.4 * (1e3) / (3e8)
# Assume AlN index of 2.08.
local_delay_per_MZI = 1e9 * 2.08 * (3e-5) / (3e8)

# Consecutive BK attempts can only be triggered after the detector dead time.
detector_dead_time = 12 # nanoseconds (value from Ian?)

# Detector dark count rate.
detector_pdark_rate_per_ns = 1e-8
# Detector efficiency.
detector_eff = 0.93

# Fidelity of two-qubit gates and measurement.
# Note that single-qubit gates can be done with high fidelity and relatively low times.
gate_fidelity = 0.98 # 0.98 is the value from Rozpedek et al, but for 2 registers,
					  # hybrid fidelity is 0.87 and trad fidelity is 0.91.
					  # gate_fidelity = 0.99 gives hybrid = 0.928, trad = 0.930.
					  # gate_fidelity = 0.999 gives hybrid = 0.98, trad = 0.95.
					  # These values assume prep_fidelity = 1.
meas_fidelity = 0.9998 # This value is for SiVs. It should be about 0.992 for NVs 
					   # (see "High fidelity spin measurement on the nitrogen vacancy center")
prep_fidelity = 0.99

# Time needed for Bell state measurement.
# Note that this time is not actually simulated (unlike link delays, which actually elapse in real time.)
# Bell state measurements do not go into the local Barrett-Kok clock cycle; they should go into the 
# network clock cycle.
BSM_time = 1e5 # nanoseconds, i.e. 100 microseconds
# We need to finish the BSM before we can reinitialize qubits, but we can do > 1 BSMs in a single clock
# cycle simultaneously.
# Therefore, add BSM_time to link_time.
# Note that we ignore the time needed for e.g. SWAP gates -- BSM_time should dominate these other
# contributions.

# Time needed for initialization (i.e. generating spin-photon entanglement).
prep_time = 6e3

# We might argue that the traditional repeater does not need local_time, but BSM_time dominates
# local_time by such a large factor that the network clock cycle (link_time) will not change much.

### COMPUTE PARAMETERS ###
def get_params(m, channel_length, duration):
	# m = number of qubits in repeater
	# 		The number of channels is int(m/2) in a hybrid repeater and m in a traditional repeater.
	# channel_length = distance between nodes and bs, in kilometers
	channel_efficiency = fixed_channel_efficiency * 10**(-channel_loss_per_km * 0.1 * channel_length)
	channel_loss = 1 - channel_efficiency
	depth_MZIs = np.ceil(np.log2(2*int(m/2))) + 1
	repeater_channel_efficiency = fixed_repeater_channel_efficiency * \
										10**(-repeater_channel_loss_per_MZI * 0.1 * depth_MZIs)
	repeater_channel_loss = 1 - repeater_channel_efficiency
	link_delay = link_delay_per_km * channel_length
	local_delay = local_delay_per_MZI * depth_MZIs
	local_time = 2.* max(2*local_delay, detector_dead_time)
	link_time = 2.* max(2*link_delay, detector_dead_time) + 10 * local_time + BSM_time + prep_time
	time_bin = 1e-2
	detector_pdark = 1e-8 * detector_dead_time
	return (atom_times, rep_times, channel_loss, duration, \
				repeater_channel_loss, noise_on_nuclear_params, link_delay, \
				link_time, local_delay, local_time, time_bin, \
				detector_pdark, detector_eff, gate_fidelity, meas_fidelity, prep_fidelity)

def hybrid_params(m, channel_length, duration):
	return (int(m/2),) + get_params(m, channel_length, duration)

def trad_params(m, channel_length, duration):
	# Still add the 10*local_time contribution -- is this fair?
	atom_times, rep_times, channel_loss, duration, \
				repeater_channel_loss, noise_on_nuclear_params, link_delay, \
				link_time, local_delay, local_time, time_bin, \
				detector_pdark, detector_eff, gate_fidelity, meas_fidelity, prep_fidelity\
				= get_params(m, channel_length, duration)
	return (m, atom_times, rep_times, channel_loss, duration, None, \
				noise_on_nuclear_params, link_delay, link_time, None, None, time_bin, \
				detector_pdark, detector_eff, gate_fidelity, meas_fidelity, prep_fidelity)

data_hybrid = []
data_trad = []
for m in num_qubits:
	data_hybrid.append([])
	data_trad.append([])
	for k in range(num_repeats):
		print('hybrid', m, k)
		result = run_hybrid(*hybrid_params(m, channel_length, duration)).data
		with open(PREFIX+'run_test2_data_hybrid_'+str(m)+'_trial_'+str(k)+'.pickle', 'wb') as fn: 
			pickle.dump(result, fn)
		data_hybrid[-1].append(result)
		print('trad', m, k)
		result = run_trad(*trad_params(m, channel_length, duration)).data
		with open(PREFIX+'run_test2_data_trad_'+str(m)+'_trial_'+str(k)+'.pickle', 'wb') as fn: 
			pickle.dump(result, fn)
		data_trad[-1].append(result)

# Save data!
with open(PREFIX+'run_test2_data_hybrid.pickle', 'wb') as fn: pickle.dump(data_hybrid, fn)
with open(PREFIX+'run_test2_data_trad.pickle', 'wb') as fn: pickle.dump(data_trad, fn)

