import netsquid as ns
import netsquid.qubits as nq
import numpy as np

from easysquid.connection import ClassicalConnection, QuantumConnection
from easysquid.qnode import QuantumNode
from easysquid.easyprotocol import EasyProtocol, ClassicalProtocol
from netsquid.protocols import TimedProtocol
from easysquid.easynetwork import EasyNetwork
from easysquid.toolbox import EasySquidException, logger
from netsquid.simutil import sim_time

from easysquid.quantumMemoryDevice import QuantumMemoryDevice, StandardMemoryDevice,\
											UniformStandardMemoryDevice
from netsquid.components import QuantumNoiseModel, FixedDelayModel

import logging

from bk8 import AtomProtocol, BeamSplitterProtocol, DetectorProtocol,\
				StateCheckProtocol, QubitLossNoiseModel
# Use BSM3 because it allows for imperfect measurements.
from BSM3 import BellStateMeasurement as BSM
from BSM3 import depolarization

np.set_printoptions(precision=2)

# Multiple channels from Alice to Bob through repeater node, arranged in hybrid fashion.
# A---N---C1--(fully bipartite)--C2---N---B
# This time, packaged nicely to only expose Alice's and Bob's scont nodes.
# Also with proper C1, C2 connections i.e. with electron and nuclear spins.

# The repeater connections need not be perfect, and we incorporate the loss model in Rozpedek
# et al for the decoherence when an electron attempt is made.
# Connections are treated as having a physical distance, which means they have an associated
# time delay as well.

# Includes minor effects: incoherence and time needed for 2-qubit gates, Bell state measurement
# 						  and state preparation.

# Note the changes to the arguments of run_simulation.

class PrintProtocol(TimedProtocol):
	def __init__(self, time_step, nodes_to_print, start_time = 0, to_combine = False):
		super().__init__(time_step, start_time = start_time)
		self.nodes_to_print = nodes_to_print
		self.to_combine = to_combine
	
	def run_protocol(self, args = None):
		if self.to_combine:
			print('nodes', [node.ID for node in self.nodes_to_print])
			with np.printoptions(threshold=np.inf): 
				print(nq.reduced_dm([node.qmemory.get_qubit(0) for node in self.nodes_to_print]))
		else:
			for node in self.nodes_to_print:
				print('node', node.ID, '\n', node.qmemory.get_qubit(0).qstate.dm)

class SourceProtocol(AtomProtocol):
	''' Protocol for source nodes. '''
	def __init__(self, time_step, node, connection, test_conn, to_correct = False, prep_fidelity = 1.):
		super().__init__(time_step, node, connection, test_conn, to_run = False, \
						 to_correct = to_correct, prep_fidelity = prep_fidelity)
		self.scont_conn = None
	
	def start_BK(self, args = None):
		# Check the message.
		[msg,], _ = self.scont_conn.get_as(self.myID)
		#print(msg, self.myID, 'source')
		if msg == 'start':
			# Start the Barrett-Kok protocol by sending photons down to the beamsplitter.
			self.run_protocol_inner()
	
	def set_scont_conn(self, conn_to_scont):
		# Set up a classical connection to the source control.
		self.scont_conn = conn_to_scont

class BellProtocol(AtomProtocol):
	''' Protocol for repeater atoms. '''
	def __init__(self, time_step, node, connection, test_conn, \
								to_correct = False, to_correct_BK_BSM = False, to_print = False,
								noise_on_nuclear = None, \
								gate_fidelity = 1., meas_fidelity = 1., prep_fidelity = 1.):
		# to_correct is for the AtomProtocol Barrett-Kok (i.e. with the source node).
		# to_correct_BK_BSM is whether to correct phases in the BK_BSM protocol.
		# noise_on_nuclear is the noise to be applied on the nuclear spin every time the electron spin is
		#	used; it comprises a dephasing and depolarization, see Rozpedek et al.
		# gate_fidelity is the fidelity with which a single 2-qubit gate can be applied.
		# meas_fidelity is the fidelity with which the electron spin can be be measured -- affects
		# 	the fidelity of Bell state measurements.
		super().__init__(time_step, node, connection, test_conn, to_run = False, \
						 to_correct = to_correct, prep_fidelity = prep_fidelity)
		# Note that the quantum memory should have 2 atoms now:
		#	electron spin (active) = index 0
		# 	nuclear spin (storage) = index 1
		self.repeater_conn = None # classical connection to repeater control
		self.repeater_bs_conn = None # quantum connection to repeater beamsplitter
		self.test_repeater_bs_conn = None # classical connection to repeater beamsplitter
		self.num_electron_attempts = 0 # number of photons sent by the electron spin before a
									   # successful run of BK_BSM is achieved
									   # This is important because it may affect the nuclear spin.

		# To keep track of which detector observed the photon.
		self.detector_order_BK_BSM = None
		self.to_correct_BK_BSM = to_correct_BK_BSM

		# Whether to print info to console.
		self.to_print = to_print

		# Noise to be applied on the nuclear spin.
		# noise_on_nuclear takes a single qubit as input, and modifies the state of the qubit in place.
		self.noise_on_nuclear = noise_on_nuclear
		# Fidelity of single 2-qubit gate.
		self.gate_fidelity = gate_fidelity
		# Fidelity after depolarization due to electron spin measurements.
		self.meas_fidelity = meas_fidelity
		# Depolarization channel for the SWAP gate.
		# This is for the situation where the electron spin is moved _into_ the nuclear spin,
		# so there is nothing in the electron spin after the nuclear spin is moved.
		# Hence, we only need a one-qubit depolarization channel.
		self.gate_depol = depolarization(1, gate_fidelity**2)

	def apply_noise_on_nuclear(self):
		if self.noise_on_nuclear is not None:
			nuclear_spin = self.nodememory.get_qubit(1)
			if nuclear_spin is not None:
				self.noise_on_nuclear(nuclear_spin)
		
	def send_photon(self):
		# Send photon down the connection.
		super().send_photon()
		# The quantum memory also has a nuclear spin now, so apply the appropriate noise to the 
		# nuclear spin.
		self.apply_noise_on_nuclear()
	
	def start_BK(self, args = None):
		# Called when the repeater node should do Barrett-Kok with the source nodes.
		# Also called when the two repeater nodes should do a BSM so that Alice and Bob share 
		# entanglement. In the latter case, the repeater control sends a message down to the 
		# repeater nodes to start the BK_BSM protocol.

		# Check the message.
		[msg,], _ = self.repeater_conn.get_as(self.myID)
		#print(msg, self.myID, 'bell')
		if msg == 'start':
			# Start the Barrett-Kok protocol by sending photons down to the beamsplitter.
			self.run_protocol_inner()
		elif msg == 'BK_BSM_start':
			# Just a quick check that the repeater node is in the right status.
			assert self.stage == 3
			# First move the electron spin to the nuclear spin.
			self.move_qubit(0, 1)
			# Note that moving the electron spin to the nuclear spin is a SWAP operation, which
			# requires two 2-qubit gates.
			# Simulate errors using a depolarizing channel.
			self.gate_depol([self.nodememory.get_qubit(1)])
			# The nuclear qubit just got started, so restart the electron counter.
			self.num_electron_attempts = 0
			self.stage = 4
			self.nodememory.release_qubit(0) # just in case
			self.BK_BSM_inner()

	def set_repeater_conn(self, conn_to_repeater_control):
		# Set up a connection to the repeater control which will collect entanglement information
		# from the atoms.
		# The connection should be a ClassicalConnection.
		self.repeater_conn = conn_to_repeater_control

	def verification(self, args = None):
		super().verification(args = None)
		if self.stage == 3 and self.repeater_conn is not None:
			#print('success', self.myID, self.repeater_conn)
			# If success, inform the repeater control.
			self.repeater_conn.put_from(self.myID, data = ["success",])

	def set_repeater_bs_conn(self, conn_to_repeater_beamsplitter, test_conn_to_rep_bs):
		# Should be a QuantumConnection.
		self.repeater_bs_conn = conn_to_repeater_beamsplitter
		self.test_repeater_bs_conn = test_conn_to_rep_bs
	
	def BK_BSM_inner(self):
		# Start the repeater's internal Barrett-Kok protocol by sending photons down to the
		# repeater beamsplitter.
		# Repeat run_protocol_inner, but for self.repeater_bs_conn instead of self.conn.
		# Also define stages appropriately.
		# stage = 4: starting repeater BK_BSM protocol
		# stage = 5: Alice/Bob is entangled to the nuclear spin; electron spin just sent
		#			 out the first photon to the repeater beamsplitter, waiting for "failed"
		#			 or "next stage" instruction
		# stage = 6: Alice/Bob is still entangled to the nuclear spin; electron spin just
		#			 sent out the second photon, waiting for "failed" or "succeeded" signal
		# stage = 7: Alice/Bob is still entangled to the nuclear spin; succeeded!
		# stage = 8: BSM completed between electron and nuclear spin; Alice/Bob now entangled
		#			 to each other
		# stage = 9: BK_BSM failed, waiting for next trigger
		# Reuse self.atom and self.photon??
		self.set_state()
		# Need to replace self.send_photon(), 
		# since that sends it down the connection with Alice/Bob.
		self.repeater_bs_conn.put_from(self.myID, data = [self.photon])
		# The electron spin has sent a photon, so increment the counter.
		self.apply_noise_on_nuclear()
		self.num_electron_attempts += 1
		self.stage = 5

	def BK_BSM_verification(self, args = None):
		# Callback function when first and second phases of Barrett-Kok are successful.
		[msg,], _ = self.test_repeater_bs_conn.get_as(self.myID)
		if self.to_print: print(msg, self.myID)
		if msg[:-2] == 'BK_BSM_success' and self.stage == 5:
			# Verify the state.
			self.nodememory.operate(ns.X, 0)
			self.photon, = nq.create_qubits(1)
			# Entangle self.atom and self.photon.
			# Again, note that self.atom is reused here for the repeater operation.
			nq.operate([self.atom, self.photon], self.permute34)
			# Send photon.
			self.repeater_bs_conn.put_from(self.myID, data = [self.photon])
			self.apply_noise_on_nuclear()
			self.num_electron_attempts += 1
			# Keep track of which detector observed the photon.
			self.detector_order_BK_BSM = msg[-2:]
			self.stage = 6
			# Note that if the connections are perfect, the verification stage (i.e. the second round
			# of Barrett-Kok) will most likely succeed.
		elif msg[:-2] == 'BK_BSM_success' and self.stage == 6:
			# Correct BK_BSM phase first.
			if self.to_correct_BK_BSM and msg[-2:] != self.detector_order_BK_BSM:
				self.nodememory.operate(ns.Z, 0)
			self.detector_order_BK_BSM = None
			# Print BK_BSM state.
			#print(self.myID, self.nodememory.get_qubit(0).qstate)
			# Notify beamsplitter that the correction has been made, in preparation for BSM.
			self.test_repeater_bs_conn.put_from(self.myID, data = ['BK_BSM_readyBSM',])
			self.stage = 7
		elif msg == 'BK_BSM_doBSM' and self.stage == 7:
			# Perform a BSM now.
			# Note that the fidelity of a BSM has two contributions:
			# one from the measurement of the electron spin (x2)
			# and one from the SWAP operation. The SWAP operation is really two 2-qubit gates, so
			# we need gate_fidelity**2 as our cumulative fidelity (see Rozpedek et al).
			BSMoutcome = BSM(self.nodememory.get_qubit(0), self.nodememory.get_qubit(1), \
								self.gate_fidelity**2, self.meas_fidelity)
			# TODO: BSM should also return some results? not sure if relevant;
			# could always pass the results down self.repeater_conn.
			self.repeater_conn.put_from(self.myID, data = \
									[("BK_BSM_success", self.num_electron_attempts, \
											(BSMoutcome[0][0][0], BSMoutcome[0][1][0])),])
			self.stage = 8
			# Release the electron and nuclear spins.
			self.nodememory.release_qubit(0)
			self.nodememory.release_qubit(1)
		else:
			self.stage = 9 # failure, so try again

	def move_qubit(self, index_from, index_to):
		# Moves qubit from index_from to index_to in the quantum memory of self.node.
		self.nodememory.add_qubit(self.nodememory.get_qubit(index_from), index_to)
		self.nodememory.release_qubit(index_from)

	def run_protocol(self):
		# Need to make sure that BK_BSM is attempted, if possible.
		# Here, self.to_run is useless.
		# Remember to register the node protocol and start the node!!
		if self.stage == 9:
			# After failure, clear the electron spin.
			self.nodememory.release_qubit(0)
			self.BK_BSM_inner()	
	
class RepeaterControlProtocol(TimedProtocol):
	''' Protocol for repeater control. '''
	def __init__(self, time_step, node, rep1, conn1, rep2, conn2, to_print = False):
		# conn1, rep1 = list of connections from repeater nodes 1 (rep1) to the repeater control
		# conn2, rep2 = list of connections from repeater nodes 2 (rep2) to the repeater control
		# Need to have access to the repeater nodes so that the Bell state measurement can be performed.
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.conn1 = conn1
		self.rep1 = rep1
		self.conn2 = conn2
		self.rep2 = rep2

		# Connection to sconts.
		self.scont_conn = [None, None]

		# Lists of source nodes on either side.
		# These are connected to self.rep1 and self.rep2 through self.conn1 and self.conn2 respectively.
		# Could be useful for data collection.
		self.source1 = None
		self.source2 = None
		
		# Status = 1 if the repeater node is entangled with Alice/Bob
		self.status1 = [0]*len(conn1)
		self.status2 = [0]*len(conn2)

		# Data collection.
		self.to_collect_data = True
		# self.times1[i] = sim_time when the (i+1)-th entanglement was achieved.
		# Similarly for self.times2[i].
		self.times1 = []
		self.times2 = []
		# self.key_times[i] = sim_time at the (i+1)-th time Alice and Bob share an entangled qubit.
		self.key_times = []
		# self.key_dm[i] = density matrix of the (i+1)-th time Alice and Bob share an entangled qubit.
		self.key_dm = []
		# All data, to be mutated.
		self.data = [self.times1, self.times2, self.key_times, self.key_dm]

		# ClassicalConnection to repeater beamsplitter.
		self.repeater_bs_conn = None
		# Key-value pairs of repeater nodes we are waiting for.
		# In particular, self.BK_BSM_wait1[index1] = index2 and self.BK_BSM_wait2[index2] = index1.
		self.BK_BSM_wait1 = dict()
		self.BK_BSM_wait2 = dict()

		# Whether to print info to console.
		self.to_print = to_print

	def is_entangled(self, side, index, args = None):
		# Method to run when the repeater nodes inform the control that entanglement has been
		# established, either with the source node (Alice/Bob) or with the other repeater nodes.
		# The side can be determined by setting up the callback-channel linkage appropriately.
		if side == 1:
			[data,], _ = self.conn1[index].get_as(self.myID)
			# Entanglement with source node.
			if data == "success": 
				self.status1[index] = 1
				if self.to_collect_data: self.times1.append(sim_time())
			# Entanglement with other repeater node, with corresponding BSM measurement performed.
			# Note that the number of failures is encoded in data[1] as well.
			elif isinstance(data, tuple) and data[0] == "BK_BSM_success":
				# Note: we should not pop from self.BK_BSM_wait1 yet, because self.BK_BSM_wait1(index)
				# = None implies that the repeater node has successfully achieved entanglement with the
				# source node but has not yet done the BSM. Instead, our repeater node is waiting for
				# the companion repeater node to do the BSM.
				# What we will do is to "mark" the repeater node as done using the string 
				# "BK_BSM_success". The companion node should check that this node is done later.
				other_index, start_time = self.BK_BSM_wait1[index] # extract companion node
				self.BK_BSM_wait1[index] = ("BK_BSM_success", data[2]) # set to placeholder,
																	   # also record BSM results
				# If the companion node is already done, declare the result and reset the repeater
				# nodes' statuses.
				if self.BK_BSM_wait2[other_index][0] == "BK_BSM_success":
					self.BK_BSM_wait1.pop(index)
					_, other_results = self.BK_BSM_wait2.pop(other_index)
					self.BScomplete(index, other_index, start_time = start_time, \
										num_electron_attempts = data[1], \
										results = (data[2], other_results))
		elif side == 2:
			[data,], _ = self.conn2[index].get_as(self.myID)
			if data == "success": 
				self.status2[index] = 1
				if self.to_collect_data: self.times2.append(sim_time())
			elif isinstance(data, tuple) and data[0] == "BK_BSM_success":
				other_index, start_time = self.BK_BSM_wait2[index] # extract companion node
				self.BK_BSM_wait2[index] = ("BK_BSM_success", data[2]) # set to placeholder
				if self.BK_BSM_wait1[other_index][0] == "BK_BSM_success":
					self.BK_BSM_wait2.pop(index)
					_, other_results = self.BK_BSM_wait1.pop(other_index)
					self.BScomplete(other_index, index, start_time = start_time, \
										num_electron_attempts = data[1],\
										results = (other_results, data[2])) 
					# note the order - always (side 1, side 2)
		# If status1 and status2 are both 1, then we can perform a Bell state measurement!
		# It is appropriate to do the BSM here because the status can only change when a message is sent.
		if sum(self.status1) * sum(self.status2) > 0:
			# Step through each side and perform as many BSMs as possible.
			counter1, counter2 = 0, 0
			# Keep incrementing counter1, counter2 until we find repeater nodes with status 1
			# that are waiting to be matched.
			while True:
				# Note: we do not want to re-initialize the BSM sequence for repeater nodes that
				# are in the process of running BK_BSM.
				# These repeater nodes will still have status 1, but should not be matched.
				while self.status1[counter1] == 0 or self.BK_BSM_wait1.get(counter1) is not None:
					counter1 += 1
					if counter1 >= len(self.status1): break
				while self.status2[counter2] == 0 or self.BK_BSM_wait2.get(counter2) is not None: 
					counter2 += 1
					if counter2 >= len(self.status2): break
				if counter1 >= len(self.status1) or counter2 >= len(self.status2): break
				self.BSmeasure(counter1, counter2, sim_time())
	
	def BSmeasure(self, index1, index2, curr_time = None):
		# Pass the current time through BSmeasure so that we can keep track of when the BK_BSM
		# protocol was started. This will be useful when we are recording the start and end times
		# of BK_BSM after entanglement was successfully achieved.
		
		# Perform a Bell state measurement on the repeater nodes using the following process:
		#	1. move repeater qubits from the electron spins to the nuclear spins
		#	2. establish entanglement between electron spins
		#	3. perform two BSMs between electron and nuclear spins
		# These will be done at the repeater nodes directly.
		# All repeater nodes will be connected to a single repeater beamsplitter in this setup.
		# The repeater control will tell the repeater beamsplitter which nodes should be 
		# combined appropriately.
		self.conn1[index1].put_from(self.myID, data = ['BK_BSM_start',])
		self.conn2[index2].put_from(self.myID, data = ['BK_BSM_start',])
		self.repeater_bs_conn.put_from(self.myID, data = [('add', index1, index2),])
		# After this is done, the repeater control should wait for both repeater nodes to 
		# declare a successful BK_BSM.
		# Add the pair of repeater nodes to the set.
		# TODO: make this exception better.
		if self.BK_BSM_wait1.get(index1) is not None or self.BK_BSM_wait2.get(index2) is not None: 
			raise Exception
		# Also encode the current time into the dictionaries.
		self.BK_BSM_wait1[index1] = (index2, curr_time)
		self.BK_BSM_wait2[index2] = (index1, curr_time)
				
	def BScomplete(self, counter1, counter2, start_time = None, num_electron_attempts = None,\
					results = None):
		# start_time is passed from BSmeasure through the repeater nodes.
		# num_electron_attempts is passed from the repeater nodes.
		# results = BSM results from side 1 and side 2.
		if self.to_print: print('BSM results', results)

		# Perform unitaries based on BSM results.
		# Requires access to source nodes.
		
		# Idea: (0, 0) = (01+10) is what we want, so no flip required
		# 		(0, 1) = (01-10) requires a parity flip (Z)
		#		(1, 0) = (00+11) requires a population flip (X)
		#		(1, 1) = (00-11) requires a population and parity flip (Z, X - order doesn't matter
		#																up to a phase)
		# 2 population flips (one on each side) cancel out,
		# 2 parity flips (one on each side) cancel out.

		if results[0][0] == 1:
			nq.operate(self.source1[counter1].qmemory.get_qubit(0), ns.X)
		if results[0][1] == 1:
			nq.operate(self.source1[counter1].qmemory.get_qubit(0), ns.Z)
			
		if results[1][0] == 1:
			nq.operate(self.source2[counter2].qmemory.get_qubit(0), ns.X)
		if results[1][1] == 1:
			nq.operate(self.source2[counter2].qmemory.get_qubit(0), ns.Z)

		# The two companion repeater nodes have successfully performed a BK_BSM.
		# Restore statuses to normal, i.e. set status back to zero.
		self.status1[counter1] = 0
		self.status2[counter2] = 0
		# Record the BSM time and other information that was provided.
		# Note that this line can be converted to a write-to-file operation if memory is limited.
		if self.to_collect_data: self.key_times.append((sim_time(), start_time, num_electron_attempts))
		# Note: if multiple messages are sent down the same channel in the
		#  same timestep, then the messages will be concatenated in a list.
		#  Hence, better to group messages in tuples, as multiple BSMs may mean
		#  multiple messages sent at the same time.
		self.scont_conn[0].put_from(self.myID, data = [('bsm', counter1)])
		self.scont_conn[1].put_from(self.myID, data = [('bsm', counter2)])
		self.repeater_bs_conn.put_from(self.myID, data = [('remove', counter1, counter2),])
		# Record density matrix of the source nodes, if possible.
		if self.to_collect_data and self.source1 is not None and self.source2 is not None:
			self.key_dm.append(nq.reduced_dm([self.source1[counter1].qmemory.get_qubit(0), \
							 self.source2[counter2].qmemory.get_qubit(0)]))
			# TODO: print statement
			#print(results, np.abs(np.sum(self.key_dm[-1][1:3, 1:3]))/2, \
			#			[self.data[i][len(self.key_dm)-1] for i in range(3)])
			#print(self.key_dm[-1])
	
	def set_scont_conn(self, conn1, conn2):
		# Set connections to sconts.
		self.scont_conn = [conn1, conn2]

	def set_sources(self, source1, source2):
		# Set the sources that rep1 and rep2 are connected to.
		self.source1, self.source2 = source1, source2

	def set_repeater_bs_conn(self, conn):
		self.repeater_bs_conn = conn

	def run_protocol(self):
		# Notify source controls about the successful entanglements.
		# Note that messages sent down the scont-to-repeater control connection must be
		# bundled in tuples, because the same connection can be used to send multiple BSM
		# results simultaneously.
		self.scont_conn[0].put_from(self.myID, data = [('status update', self.status1),])
		self.scont_conn[1].put_from(self.myID, data = [('status update', self.status2),])
		# Notify repeater nodes that need to start BK.
		for i in range(len(self.status1)):
			if self.status1[i] == 0: # remember, NOT "if i == 0"
				self.conn1[i].put_from(self.myID, data = ['start'])
		for i in range(len(self.status2)):
			if self.status2[i] == 0:
				self.conn2[i].put_from(self.myID, data = ['start'])
	
class ScontProtocol(TimedProtocol):
	''' Protocol for source control.'''
	def __init__(self, time_step, node, sources, conns, to_print = False):
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.sources = sources
		self.conns = conns

		# Connection to repeater control.
		self.rep_conn = None

		# Status = 1 if the source node is entangled with the repeater.
		self.status = [0]*len(conns)

		self.to_print = to_print
	
	def set_repeater_conn(self, rep_conn):
		self.rep_conn = rep_conn
	
	def msg_from_repeater(self, args = None):
		msgs, _ = self.rep_conn.get_as(self.myID)
		# Note that messages from the scont-to-repeater control connection come in tuples of 
		# (msg, data), as multiple BSMs may be performed in the same time step.
		for msg, data in msgs:
			if msg == 'status update':
				# Will get a status update at every time step.
				self.status = data
				# Get source nodes to start Barrett-Kok.
				for i in range(len(self.conns)):
					if self.status[i] == 0:
						self.conns[i].put_from(self.myID, data = ['start'])
					# If the source node is waiting for entanglement on the other side,
					# don't do anything.
			elif msg == 'bsm':
				if self.to_print: 
					print('bsm', self.myID, data, sim_time())
					print(self.sources[data].qmemory.get_qubit(0).qstate)
				pass # don't need to make measurements for now

class RepeaterBSProtocol(TimedProtocol):
	''' 
	Protocol for repeater beam splitter, which should always be paired with a repeater control node. 
	Main purpose: reroute connections between companion repeater nodes to the same beamsplitter/
	detector pairs.
	'''
	def __init__(self, time_step, node, conn_to_rep_control, conn1, conn2, test_conn1, test_conn2):
		super().__init__(time_step, node = node, connection = None)

		# Connection to the repeater control.
		self.conn_to_rep_control = conn_to_rep_control
		# May not be used for outgoing messages?
		# QuantumConnection to/from the repeater nodes.
		self.conn1 = conn1
		self.conn2 = conn2
		# ClassicalConnection to/from the repeater nodes.
		# e.g. test_conn1[i] = connection from index1 = i
		self.test_conn1 = test_conn1
		self.test_conn2 = test_conn2

		# Naive approach: one beamsplitter for each pair of repeater nodes.
		# Connections to/from the beamsplitter.
		self.repconn1 = None
		self.repconn2 = None
		self.test_repconn1 = None
		self.test_repconn2 = None

		# Keep track of which pairs of nodes are active.
		self.from1to2 = dict()
		self.from2to1 = dict()

		# Keep track of the nodes that we are waiting for the 'BK_BSM_readyBSM' message from.
		# Key (side, index) -> value (index1, index2).
		# If (side, index) is received but not in self.waiting, then set up the companion (side, index).
		# If (side, index) is in self.waiting, then send 'BK_BSM_doBSM' message to both indices.
		self.waiting = dict()

		self.myID = self.node.nodeID
	
	def set_repconn(self, repconn1, repconn2, test_repconn1, test_repconn2):
		# repconn1[i][j] = connection if index1 = i and index2 = j are trying to connect, for the 
		#					first qubit
		# repconn2[i][j] = connection if index1 = i and index2 = j are trying to connect, but for
		#					the second qubit
		self.repconn1 = repconn1
		self.repconn2 = repconn2
		self.test_repconn1 = test_repconn1
		self.test_repconn2 = test_repconn2
	
	def from_rep_control(self, args = None):
		# Callback when a new pair of nodes is sent from the repeater control.
		data, _ = self.conn_to_rep_control.get_as(self.myID)
		for inst, index1, index2 in data:
			if inst == 'add':
				self.from1to2[index1] = index2
				self.from2to1[index2] = index1
			elif inst == 'remove':
				self.from1to2.pop(index1)
				self.from2to1.pop(index2)
				# Also erase the waiting dictionary.
				if self.waiting.get((1, index1)) is not None: self.waiting.pop((1, index1))
				if self.waiting.get((2, index2)) is not None: self.waiting.pop((2, index2))
	
	def from_rep_nodes(self, side, index, args = None):
		# Route data from the repeater nodes to the appropriate beamsplitters.
		if side == 1:
			data, _ = self.conn1[index].get_as(self.myID)
			# Find out who to route it to.
			other_index = self.from1to2[index]
			self.repconn1[index][other_index].put_from(self.myID, data = data)
		elif side == 2:
			data, _ = self.conn2[index].get_as(self.myID)
			other_index = self.from2to1[index]
			self.repconn2[other_index][index].put_from(self.myID, data = data)

	def from_rep_nodes_readyBSM(self, side, index, args = None):
		# Check the incoming message from repeater nodes to see if they have performed the required
		# unitaries after BK.
		if side == 1:
			[msg,], _ = self.test_conn1[index].get_as(self.myID)
			if msg == 'BK_BSM_readyBSM':
				other_index = self.from1to2[index]
				if (1, index) not in self.waiting:
					self.waiting[(2, other_index)] = (index, other_index)
				else:
					self.test_conn1[index].put_from(self.myID, data = ['BK_BSM_doBSM',])
					self.test_conn2[other_index].put_from(self.myID, data = ['BK_BSM_doBSM',])
					self.waiting.pop((1, index))
		elif side == 2:
			[msg,], _ = self.test_conn2[index].get_as(self.myID)
			if msg == 'BK_BSM_readyBSM':
				other_index = self.from2to1[index]
				if (2, index) not in self.waiting:
					self.waiting[(1, other_index)] = (other_index, index)
				else:
					self.test_conn2[index].put_from(self.myID, data = ['BK_BSM_doBSM',])
					self.test_conn1[other_index].put_from(self.myID, data = ['BK_BSM_doBSM',])
					self.waiting.pop((2, index))
			
	def from_bs(self, side, index, args = None):
		# Route data from the beamsplitters to the appropriate repeater nodes.
		# Note that the repeater nodes respond to messages with the 'BK_BSM_' prefix.
		if side == 1:
			[msg,], _ = self.test_repconn1[index][self.from1to2[index]].get_as(self.myID)
			self.test_conn1[index].put_from(self.myID, data = ['BK_BSM_'+msg,])
		elif side == 2:
			[msg,], _ = self.test_repconn2[self.from2to1[index]][index].get_as(self.myID)
			self.test_conn2[index].put_from(self.myID, data = ['BK_BSM_'+msg,])
		# Note that the repeater nodes will inform the repeater control of success/failure directly.

# Functions needed for runtime behavior.
if True:
	def connect_by_hybrid_repeater(num_channels, scont1, scont2, repeater_control, repeater_bs,  \
							make_BK, make_scont_conn, make_rep_control_conn,\
							make_rep_bs_conn):
		# Connects the source controls scont1, scont2 via the repeater control rep_control.
		# Inputs:
		#	num_channels = (int) number of channels connected in the hybrid/MIT fashion
		#					Note: if num_channels is a 2-tuple of ints, then num_channels[0]
		#					is the number of repeater nodes on Alice's side and num_channels[1]
		#					is that number on Bob's side.
		#	scont1, scont2, rep_control = QuantumNodes
		#	repeater_bs = QuantumNode that routes qubits from repeater nodes on to appropriate
		#				  beam splitters for internal (i.e. in-repeater) BSM
		# 	make_BK = function that returns [[sender_node, rep_node], [sender_prot, rep_prot]]
		#			  i.e. makes Barret-Kok connections
		#			  Note that the sender/repeater protocols are responsive, so they don't have
		#			  to do anything without being prompted by a message.
		#	make_scont_conn = function that takes (source nodes, source node protocols, scont) and
		#					   connects source nodes to scont nodes, returning [scont_prot, conns]
		#					   Note that connect_to_scont should automatically generate scont
		#					   protocols.
		#					   Also note that the scont protocol is responsive. Like the sender/repeater
		#					   protocols above, this means that they don't need to be added as a node
		#					   protocol. They are effectively wrappers over callback functions.
		#	make_rep_control_conn = function that takes (rep1, rep1_proto, rep2, rep2_proto, 
		#							 rep_control) and connects repeater nodes to repeater controls, 
		#							 returning [rep_control_prot, conns1, conns2]
		#							 Note that the repeater control protocol is NOT purely responsive,
		#							 as it has to initiate the Barrett-Kok sequence at every specified
		#							 time interval. Therefore, connect_to_rep_control should add
		#							 control_prot as a node protocol.
		#	make_rep_bs_conn = function that takes ((rep1, rep1_proto, rep2, rep2_proto), 
		#						repeater_control, rep_control_prot, repeater_bs)
		#						and connects repeater nodes to the repeater beamsplitter
		# Returns:
		#	rep_control_prot, scont1_prot, scont2_prot

		if isinstance(num_channels, int):
			num_channels = (num_channels, num_channels)

		# Construct source nodes and repeater nodes (which actually contain atoms).
		source1 = [] # Alice
		source1_proto = []
		rep1 = []
		rep1_proto = [] # repeater protocols
		source2 = [] # Bob
		source2_proto = []
		rep2 = []
		rep2_proto = [] # repeater protocols

		for i in range(num_channels[0]):
			[a, b], [c, d] = make_BK() 
			source1.append(a) # source node
			rep1.append(b) # repeater node
			source1_proto.append(c)
			rep1_proto.append(d) # repeater protocol
			
		for i in range(num_channels[1]):
			[a, b], [c, d] = make_BK(to_correct_BK_BSM = True)
			source2.append(a) # source node
			rep2.append(b) # repeater node
			source2_proto.append(c)
			rep2_proto.append(d) # repeater protocol
			
		scont1_prot, _ = make_scont_conn(source1, source1_proto, scont1)
		scont2_prot, _ = make_scont_conn(source2, source2_proto, scont2)

		control_prot, _, _ = make_rep_control_conn(rep1, rep1_proto, rep2, rep2_proto, \
														repeater_control) 

		r2sc1 = ClassicalConnection(repeater_control, scont1)
		r2sc2 = ClassicalConnection(repeater_control, scont2)

		# Set connections.
		scont1_prot.set_repeater_conn(r2sc1)
		scont2_prot.set_repeater_conn(r2sc2)
		control_prot.set_scont_conn(r2sc1, r2sc2)
		r2sc1.register_handler(scont1.nodeID, scont1_prot.msg_from_repeater)
		r2sc2.register_handler(scont2.nodeID, scont2_prot.msg_from_repeater)

		# Link sources to repeater control for data collection.
		control_prot.set_sources(source1, source2)

		rep_bs_prot = make_rep_bs_conn((rep1, rep1_proto, rep2, rep2_proto), repeater_control,\
						control_prot, repeater_bs)

		# If the repeater_bs connections are not made in this functionk, we also need to return 
		# repeater nodes and protocols; necessary for establishing repeater beamsplitters
		# later.	
		# Since we are making repeater_bs_connections here, this is not necessary.
		return control_prot, scont1_prot, scont2_prot

	def create_BK(index_gen, make_atom_memory, make_rep_memory, node_to_bs, bs_to_det, make_atom_prot, \
					make_rep_prot, make_det_prot, make_bs_prot, to_correct_BK_BSM = False,\
					classical_delay = 0.):
		# Inputs:
		#	 index_gen = generator for node indices
		#	 make_atom_memory = function that takes the atom name and index, and returns a MemoryDevice
		#				   (could be noisy)
		#	 	Example:
		#		make_memory = lambda x, y: UniformStandardMemoryDevice(x, y, T2 = 0.01)
		#	 make_rep_memory = similar as make_atom_memory, but with two spins
		#	 node_to_bs = function that takes the sender node and the beamsplitter node, and
		#				   returns a QuantumConnection between the two
		#	 	Example: 
		#		node_to_bs = lambda x, y: QuantumConnection(x, y, noise_model = QubitLossNoiseModel(0.1))
		#	 bs_to_det = function that takes the beamsplitter node and the detector node, and
		#				  returns a QuantumConnection between the two
		#				  Note: dark counts can be implemented here, with reverse amplitude damping.
		#	 make_atom_prot, make_rep_prot, make_det_prot = 
		#				  functions that take (node, QuantumConnection, ClassicalConnection) and
		#				  returns the appropriate protocol
		#		Example:
		#		make_rep_prot = lambda x, y, z: BellProtocol(x, y, z)
		#		Note on BellProtocol: we only add the repeater connection later.
		#	 make_bs_prot = [BeamSplitterProtocol.__init__, StateCheckProtocol.__init__]
		#	 to_correct_BK_BSM = True if the repeater node should correct for phase when performing
		#						 BK_BSM, False otherwise
		#						 Note: in the source-repeater Barrett-Kok protocol, the source node
		#						 always corrects for phase.
		#	classical_delay = time delay (in ns) for connection between nodes and beamsplitters
		# Returns:
		#	[sender_node, rep_node], [sender_prot, rep_prot]

		# Source i.e. Alice or Bob.
		index = index_gen.__next__()
		sender_atom = make_atom_memory("atom"+str(index), 1)
		sender_node = QuantumNode("source"+str(index), index, memDevice = sender_atom)
		# Repeater i.e. Charlie.
		index = index_gen.__next__()
		# Repeater nodes should have two atoms, one electron (0) and one nuclear (1) spin.
		rep_atom = make_rep_memory("atom"+str(index), 2)
		rep_node = QuantumNode("source"+str(index), index, memDevice = rep_atom)
		# Detectors.
		index = index_gen.__next__()
		detector1 = QuantumNode("detector"+str(index), index)
		index = index_gen.__next__()
		detector2 = QuantumNode("detector"+str(index), index)
		# Beamsplitter.
		index = index_gen.__next__()
		beamsplitter = QuantumNode("beamsplitter"+str(index), index)
		# Quantum connections from node to beamsplitter.
		conn1 = node_to_bs(sender_node, beamsplitter)
		conn2 = node_to_bs(rep_node, beamsplitter)
		# Quantum connections from beamsplitter to detectors.
		conn3 = bs_to_det(beamsplitter, detector1)
		conn4 = bs_to_det(beamsplitter, detector2)
		# Classical connections from detectors to beamsplitter to nodes.
		test_conn1 = ClassicalConnection(detector1, beamsplitter)
		test_conn2 = ClassicalConnection(detector2, beamsplitter)
		test_conn3 = ClassicalConnection(beamsplitter, sender_node, \
											delay_model = FixedDelayModel(delay=classical_delay))
		test_conn4 = ClassicalConnection(beamsplitter, rep_node, \
											delay_model = FixedDelayModel(delay=classical_delay))
		# Set up protocols at nodes, detectors and beamsplitter.
		# Note that the atom/repeater protocol must have a "verification" method as the callback.
		proto1 = make_atom_prot(sender_node, conn1, test_conn3, to_correct = True)
		proto2 = make_rep_prot(rep_node, conn2, test_conn4, to_correct_BK_BSM = to_correct_BK_BSM)
		proto3 = make_det_prot(detector1, conn3, test_conn1)
		proto4 = make_det_prot(detector2, conn4, test_conn2)
		BSproto = make_bs_prot[0](beamsplitter, [conn1, conn2], [conn3, conn4])
		SCproto = make_bs_prot[1](beamsplitter, [test_conn1, test_conn2], \
									[test_conn3, test_conn4])
		# Set up handlers.
		# setup_connection does things automatically, but we can also register handlers manually
		# especially for multi-connection nodes.
		sender_node.setup_connection(conn1, [proto1])
		rep_node.setup_connection(conn2, [proto2])
		test_conn3.register_handler(sender_node.nodeID, proto1.verification)
		test_conn4.register_handler(rep_node.nodeID, proto2.verification)
		conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
		conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
		test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
		test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)
		detector1.setup_connection(conn3, [proto3])
		detector2.setup_connection(conn4, [proto4])
		# Note that the repeater protocol must also be added as a node protocol for the
		# repeater node: see run_protocol in e.g. BellProtocol.
		rep_node.add_node_protocol(proto2)
		rep_node.start() # Needed to refresh after failure of BK_BSM.
		# Return nodes and protocols.
		# Keep the beamsplitter + detectors hidden away in the abstraction.
		return [sender_node, rep_node], [proto1, proto2]

	def connect_to_source(source_nodes, source_protos, source_control, make_scont_prot):
		# Connects all source nodes to a central control, so as to coordinate their actions.
		# Inputs: see connect_to_repeater
		# All source protocols should have a "set_scont_conn" method to set up the connection
		# to the central control.
		# "scont" = source control
		assert len(source_nodes) == len(source_protos)
		conns = []
		for i in range(len(source_nodes)):
			conn = ClassicalConnection(source_control, source_nodes[i])
			source_protos[i].set_scont_conn(conn)
			conns.append(conn)
			conn.register_handler(source_nodes[i].nodeID, source_protos[i].start_BK)
		scont_prot = make_scont_prot(source_control, source_nodes, conns)
		return scont_prot, conns

	def connect_to_repeater(rep1, proto1, rep2, proto2, rep_control, make_control_prot):
		# Connects all repeater nodes to a central control, as in the hybrid repeater architecture.
		# Inputs:
		#	rep1 = list of repeater nodes on Alice's side
		#	proto1 = list of repeater protocols on Alice's side (same length as rep1)
		#	rep2 = list of repeater nodes on Bob's side
		#	proto2 = list of repeater protocols on Bob's side (same length as rep2)
		#	rep_control = control node
		#	make_control_prot = function that takes (rep_control, rep1, conn1, rep2, conn2) and
		#						returns a protocol for the control node e.g. RepeaterControlProtocol
		# Note that all repeater protocols should have a set_repeater_conn method that takes a 
		# ClassicalConnection to rep_control as input, and sets up the protocol appropriately to
		# do the Bell state measurement.
		# Also note that the repeater control protocol should have an is_entangled method that is
		# run every time a node is successfully entangled.
		# We assume that the repeater control will perform the required Bell state measurement.
		
		# Set up connections between repeater nodes and repeater control.
		assert len(rep1) == len(proto1)
		assert len(rep2) == len(proto2)
		conn1 = []
		for i in range(len(rep1)):
			conn = ClassicalConnection(rep1[i], rep_control)
			proto1[i].set_repeater_conn(conn)
			conn1.append(conn)
		conn2 = []
		for i in range(len(rep2)):
			conn = ClassicalConnection(rep2[i], rep_control)
			proto2[i].set_repeater_conn(conn)
			conn2.append(conn)
		# Set up repeater control protocol.
		control_prot = make_control_prot(rep_control, rep1, conn1, rep2, conn2)
		# Set up handlers.
		for i in range(len(conn1)):
			# Note that having default arguments here is necessary: see
			# https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop. 
			func = lambda args, side = 1, index = i: control_prot.is_entangled(side, index, args)
			conn1[i].register_handler(rep1[i].nodeID, proto1[i].start_BK)
			conn1[i].register_handler(rep_control.nodeID, func)
		for i in range(len(conn2)):
			func = lambda args, side = 2, index = i: control_prot.is_entangled(side, index, args)
			conn2[i].register_handler(rep2[i].nodeID, proto2[i].start_BK)
			conn2[i].register_handler(rep_control.nodeID, func)
		rep_control.add_node_protocol(control_prot)
		return control_prot, conn1, conn2

	def connect_to_repeater_bs(index_gen, rep1, proto1, rep2, proto2, \
								rep_control, rep_control_prot, repeater_bs, \
								make_rep_bs_prot,\
								node_to_bs_rep, bs_to_bs_rep, bs_to_det_rep, \
								make_det_prot, make_bs_prot, \
								classical_rep_delay = 0.):
		# make_rep_bs_prot should take the following arguments:
		# (node, conn_to_rep_control, conn1, conn2, test_conn1, test_conn2)
		# and return a repeater beamsplitter protocol.
		# make_det_prot and make_bs_prot are as in create_BK.
		# node_to_bs_rep creates connections from repeater nodes to the repeater beamsplitter (i.e.
		# the node that routes incoming qubits appropriately).
		# bs_to_bs_rep and bs_to_det_rep create connections from repeater beamsplitter to the
		# constituent beamsplitters and from the constituent beamsplitters to the detectors respectively.
		# classical_rep_delay is the time delay (in ns) for classical signals travelling between the
		# repeater beamsplitter and the repeater node.
		# These connections could be lossless.

		# First establish connections from the repeater nodes to the repeater beamsplitter.
		conn1 = []
		conn2 = []
		test_conn1 = []
		test_conn2 = []
		for i in range(len(rep1)):
			next_conn = node_to_bs_rep(rep1[i], repeater_bs)
			next_test_conn = ClassicalConnection(repeater_bs, rep1[i],\
											delay_model = FixedDelayModel(delay=classical_rep_delay))
			proto1[i].set_repeater_bs_conn(next_conn, next_test_conn)
			conn1.append(next_conn)
			test_conn1.append(next_test_conn)
			# Handlers to register: 
			#	conn1[i] - qubits going into repeater_bs from repeater nodes
			#	test_conn1[i] - success messages going into rep1[i], handled by BK_BSM_verification
		for i in range(len(rep2)):
			next_conn = node_to_bs_rep(rep2[i], repeater_bs)
			next_test_conn = ClassicalConnection(repeater_bs, rep2[i],\
											delay_model = FixedDelayModel(delay=classical_rep_delay))
			proto2[i].set_repeater_bs_conn(next_conn, next_test_conn)
			conn2.append(next_conn)
			test_conn2.append(next_test_conn)

		# Set up a connection between rep_control and repeater_bs and create a protocol.
		rep_control_to_bs = ClassicalConnection(rep_control, repeater_bs)
		rep_bs_prot = make_rep_bs_prot(repeater_bs, rep_control_to_bs, conn1, conn2, test_conn1,\
									  test_conn2)
		# Establish handler for messages that the repeater beamsplitter will receive from the
		# repeater control.
		rep_control_to_bs.register_handler(repeater_bs.nodeID, rep_bs_prot.from_rep_control)
		# Inform repeater control about this connection.
		rep_control_prot.set_repeater_bs_conn(rep_control_to_bs)
		# Establish handlers as described above.
		for i in range(len(rep1)):
			func = lambda arg, side=1, index=i: rep_bs_prot.from_rep_nodes(side, index, arg)
			conn1[i].register_handler(repeater_bs.nodeID, func)
			test_conn1[i].register_handler(rep1[i].nodeID, proto1[i].BK_BSM_verification)
			func2 = lambda arg, side=1, index=i: rep_bs_prot.from_rep_nodes_readyBSM(side, index, arg)
			test_conn1[i].register_handler(repeater_bs.nodeID, func2)
		for i in range(len(rep2)):
			func = lambda arg, side=2, index=i: rep_bs_prot.from_rep_nodes(side, index, arg)
			conn2[i].register_handler(repeater_bs.nodeID, func)
			test_conn2[i].register_handler(rep2[i].nodeID, proto2[i].BK_BSM_verification)
			func2 = lambda arg, side=2, index=i: rep_bs_prot.from_rep_nodes_readyBSM(side, index, arg)
			test_conn2[i].register_handler(repeater_bs.nodeID, func2)

		# Now establish connections from the repeater beamsplitter to the constituent beamsplitters.
		# Note that we abstracted away the constituent beamsplitters so that only the connections
		# are relevant.
		repconn1 = []
		repconn2 = []
		test_repconn1 = []
		test_repconn2 = []
		for i in range(len(rep1)):
			repconn1.append([])
			repconn2.append([])
			test_repconn1.append([])
			test_repconn2.append([])
			for j in range(len(rep2)):
				# Create new beamsplitters, detectors etc.		
				index = index_gen.__next__()
				detector1 = QuantumNode("detector"+str(index), index)
				index = index_gen.__next__()
				detector2 = QuantumNode("detector"+str(index), index)
				# Beamsplitter.
				index = index_gen.__next__()
				beamsplitter = QuantumNode("beamsplitter"+str(index), index)
				# Quantum connections from node to beamsplitter.
				next_conn1 = bs_to_bs_rep(repeater_bs, beamsplitter)
				next_conn2 = bs_to_bs_rep(repeater_bs, beamsplitter)
				# Quantum connections from beamsplitter to detectors.
				next_conn3 = bs_to_det_rep(beamsplitter, detector1)
				next_conn4 = bs_to_det_rep(beamsplitter, detector2)
				# Classical connections from detectors to beamsplitter to nodes.
				next_test_conn1 = ClassicalConnection(detector1, beamsplitter)
				next_test_conn2 = ClassicalConnection(detector2, beamsplitter)
				next_test_conn3 = ClassicalConnection(beamsplitter, repeater_bs)
				next_test_conn4 = ClassicalConnection(beamsplitter, repeater_bs)
				# Set up protocols at detectors and constituent beamsplitter.
				# Note that the atom/repeater protocol must have a "verification" method as the callback.
				# Also note that the repeater beamsplitter does not require more protocols.
				next_proto3 = make_det_prot(detector1, next_conn3, next_test_conn1)
				next_proto4 = make_det_prot(detector2, next_conn4, next_test_conn2)
				BSproto = make_bs_prot[0](beamsplitter, [next_conn1, next_conn2], \
											[next_conn3, next_conn4])
				SCproto = make_bs_prot[1](beamsplitter, [next_test_conn1, next_test_conn2], \
											[next_test_conn3, next_test_conn4])
				
				# Register handlers, like in create_BK.
				next_conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
				next_conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
				next_test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
				next_test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)
				detector1.setup_connection(next_conn3, [next_proto3])
				detector2.setup_connection(next_conn4, [next_proto4])

				# Register handlers for repeater beamsplitter.
				func = lambda args, side=1, index = i: rep_bs_prot.from_bs(side, index, args)
				next_test_conn3.register_handler(repeater_bs.nodeID, func)
				func = lambda args, side=2, index = j: rep_bs_prot.from_bs(side, index, args)
				next_test_conn4.register_handler(repeater_bs.nodeID, func)

				# Add connections to the list.
				repconn1[i].append(next_conn1)
				repconn2[i].append(next_conn2)
				test_repconn1[i].append(next_test_conn3)
				test_repconn2[i].append(next_test_conn4)

		# Set these connections up so that the repeater beamsplitter can forward messages.
		rep_bs_prot.set_repconn(repconn1, repconn2, test_repconn1, test_repconn2)

		return rep_bs_prot

# Wrap procedure in a function.
def run_simulation(num_channels, atom_times, rep_times, channel_loss, duration = 100, \
					repeater_channel_loss = 0., noise_on_nuclear_params = None, \
					link_delay = 0., link_time = 1., local_delay = 0., local_time = 0.1, \
					time_bin = 0.01, detector_pdark = 1e-7, detector_eff = 0.93,\
					gate_fidelity = 0.999, meas_fidelity = 0.9998, prep_fidelity = 0.99):
	## Parameters:
	# num_channels = degree of parallelism (m/2 for hybrid, m for trad)
	# atom_times = [[T1, T2],] for Alice's and Bob's electron spins
	# rep_times = [[T1, T2] for electron spin, [T1, T2] for nuclear spin]
	# channel_loss = probability of losing a photon between a client node and a detector station
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
	# time_bin = temporal resolution for beamsplitters (i.e. we count photons in a time interval of 
	#				width time_bin after accounting for time of flight to the detector station)
	#			 time_bin can also be the temporal resolution of the detector -- then the detector dark
	#			 	count probability is computed in terms of time_bin
	## Detector parameters:
	# detector_pdark = dark count probability in an interval time_bin
	# detector_eff = detector efficiency
	## Minor parameters:
	# gate_fidelity = fidelity of a single 2-qubit gate (note that SWAP is two 2-qubit gates)
	# meas_fidelity = fidelity of measuring the electron spin
	## Assume symmetric links, i.e. all node-to-bs links have the same delay.
	## Also assume deterministic delay, i.e. that we have good control over timing.
	## Assume negligible loss for links within the detector station.
	## Note that the terminology is somewhat misleading: link_delay refers to the delay across an
	##	individual channel in the physical layer, not a link in the link layer. (This code was written
	##	before the name change.)

	ns.simutil.sim_reset()
	nq.set_qstate_formalism(ns.QFormalism.DM)

	# Index generator.
	def get_index():
		j = -1
		while True:
			j += 1
			yield j
	
	index_gen = get_index()
	
	# Misleading names: actually the time interval, not the rate.
	BASE_CLOCK_RATE = 10 # For SourceProtocol, DetectorProtocol, etc.,
						 # i.e. stuff that does not rely on frequent checks.
	BS_CLOCK_RATE = time_bin # For BeamSplitterProtocol and StateCheckProtocol, for whom time
						 	 # is supposedly of the essence because they require incoming photons
						 	 # to be coincident. ('supposedly' because these protocols
						 	 # do not actually run unprompted, i.e. run_protocol is not run. 
							 # Nevertheless the time step is used as the simultaneity condition.)
	REPEATER_CONTROL_RATE = link_time   # How often the repeater control sends start 
										# signals to repeater nodes
							  			# that are eligible. (For RepeaterControlProtocol, 
										# where run_protocol is run.)
	BK_BSM_RATE = local_time # How often the repeater nodes try BK_BSM after a failure. 
							 # (For BellProtocol.)

	# Noise on nuclear spin when electron spin sends a photon.
	if noise_on_nuclear_params is None:
		nna, nnb = 0, 0
	else:
		nna, nnb = noise_on_nuclear_params
	noise_on_nuclear = lambda q: nq.qubitapi.multi_operate(q, [ns.I, ns.X, ns.Y, ns.Z], \
												[1-nna-0.75*nnb, 0.25*nnb, 0.25*nnb, (nna+0.25*nnb)])

	make_atom_memory = lambda x, y: StandardMemoryDevice(x, y, decoherenceTimes = atom_times)
	make_rep_memory = lambda x, y: StandardMemoryDevice(x, y, decoherenceTimes = rep_times)
	node_to_bs = lambda x, y: QuantumConnection(x, y, noise_model = QubitLossNoiseModel(channel_loss),\
												delay_model = FixedDelayModel(delay=link_delay))
	bs_to_det = lambda x, y: QuantumConnection(x, y)
	make_atom_prot = lambda x, y, z, to_correct = False: \
							SourceProtocol(BASE_CLOCK_RATE, x, y, z, to_correct=to_correct,\
										   prep_fidelity = prep_fidelity)
	make_rep_prot = lambda x, y, z, to_correct = False, to_correct_BK_BSM = False: \
							BellProtocol(BK_BSM_RATE, x, y, z, to_correct=to_correct, \
										 to_correct_BK_BSM = to_correct_BK_BSM, \
										 noise_on_nuclear = noise_on_nuclear,\
										 gate_fidelity = gate_fidelity, meas_fidelity = meas_fidelity,\
										 prep_fidelity = prep_fidelity)
	make_det_prot = lambda x, y, z: DetectorProtocol(BASE_CLOCK_RATE, x, y, z, \
											pdark = detector_pdark, efficiency = detector_eff)
	make_bs_prot = [lambda x, y, z: BeamSplitterProtocol(BS_CLOCK_RATE, x, y, z),\
					lambda x, y, z: StateCheckProtocol(BS_CLOCK_RATE, x, y, z)]

	make_BK = lambda to_correct_BK_BSM = False, index_generator = None: \
				create_BK(index_gen if index_generator is None else index_generator, make_atom_memory, \
										make_rep_memory, node_to_bs, bs_to_det, make_atom_prot, \
										make_rep_prot, make_det_prot, make_bs_prot, to_correct_BK_BSM,\
										link_delay)

	make_scont_prot = lambda x, y, z: ScontProtocol(BASE_CLOCK_RATE, x, y, z)
	make_scont_conn = lambda x, y, z: connect_to_source(x, y, z, make_scont_prot)

	make_control_prot = lambda x, y, z, a, b: RepeaterControlProtocol(REPEATER_CONTROL_RATE, \
																		x, y, z, a, b)
	make_rep_control_conn = lambda x, y, z, a, b: connect_to_repeater(x, y, z, a, b, make_control_prot)

	next_index = index_gen.__next__()
	scont1 = QuantumNode("scont"+str(next_index), next_index)
	next_index = index_gen.__next__()
	scont2 = QuantumNode("scont"+str(next_index), next_index)
	next_index = index_gen.__next__()
	repeater_control = QuantumNode("rep_control"+str(next_index), next_index)

	# Vanilla QuantumConnections if repeater_channel_loss = 0.
	node_to_bs_rep = lambda x, y: QuantumConnection(x, y, \
									noise_model = QubitLossNoiseModel(repeater_channel_loss),\
									delay_model = FixedDelayModel(delay=local_delay))
	bs_to_bs_rep = bs_to_det
	bs_to_det_rep = bs_to_det
	make_rep_bs_prot = lambda x, y, a, b, c, d: RepeaterBSProtocol(BASE_CLOCK_RATE, x, y, a, b, c, d)
							
	next_index = index_gen.__next__()
	repeater_bs = QuantumNode("rep_bs"+str(next_index), next_index)

	# We don't want the repeater's internal detectors to keep printing messages.
	make_det_prot_rep = lambda x, y, z: DetectorProtocol(BASE_CLOCK_RATE, x, y, z, to_print = False, \
											pdark = detector_pdark, efficiency = detector_eff)

	make_rep_bs_conn = lambda rep_nodes_info, repeater_control, rep_control_prot, repeater_bs,\
						index_generator = None:\
					connect_to_repeater_bs(index_gen if index_generator is None else index_generator,\
											*rep_nodes_info, repeater_control, rep_control_prot, \
											repeater_bs, make_rep_bs_prot, node_to_bs_rep, \
											bs_to_bs_rep, bs_to_det_rep, make_det_prot_rep, \
											make_bs_prot, local_delay)
					
	rep_control_prot, scont1_prot, scont2_prot = \
						connect_by_hybrid_repeater(num_channels, scont1, scont2, \
						repeater_control, repeater_bs, make_BK, make_scont_conn, make_rep_control_conn, \
						make_rep_bs_conn)

	logger.setLevel(logging.DEBUG)

	scont1.start()
	scont2.start()
	repeater_control.start()

	ns.simutil.sim_run(duration=duration)

	return rep_control_prot

