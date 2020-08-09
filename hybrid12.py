import netsquid as ns
import netsquid.qubits as nq
import numpy as np
from collections import deque

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

from bk9 import AtomProtocol, BeamSplitterProtocol, DetectorProtocol,\
				StateCheckProtocol, QubitLossNoiseModel
# Use BSM3 because it allows for imperfect measurements.
from BSM3 import BellStateMeasurement as BSM
from BSM3 import depolarization

from repeater1 import SourceProtocol, BellProtocol 

np.set_printoptions(precision=2)

class RepeaterControlProtocol(TimedProtocol):
	''' 
	Protocol to control a hybrid repeater, also known as a router.
	Receives commands from a (supposedly omniscient) central planner.
	'''
	def __init__(self, time_step, node, rep1, conn1, rep2, conn2, planner_conn = None, to_print = False):
		# time_step should be the time interval between local BK+2BSM attempts.
		# Hybrid repeaters/routers have two banks of repeater registers, assumed to be preassigned.
		# 	These two banks of registers have side 1 facing adjacent nodes, and side 2 facing each
		# 	other. Side 2 of both banks are connected through a single repeater beamsplitter.
		# conn1 = list of connections from repeater registers 1 (rep1) to the repeater control
		# conn2 = list of connections from repeater registers 2 (rep2) to the repeater control
		# rep1, rep2 = lists of repeater registers
		# 				Need to have access to the repeater registers so that the Bell state 
		#				measurement can be performed.
		# planner_conn = connection to the central planner
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.conn1 = conn1
		self.rep1 = rep1
		self.conn2 = conn2
		self.rep2 = rep2

		self.planner_conn = planner_conn
		
		# self.status1 = first bank of repeater registers.
		# self.status2 = second bank of repeater registers.
		# Each bank has two sides.
		# The status reflects which side each qubit in a register is connected to.
		# self.status1[i][0] = side of broker qubit (0 if not entangled).
		# self.status1[i][1] = side of storage qubit (0 if not entangled).
		# 	status = 1 if the qubit connected to the adjacent (outer) register.
		# 	status = 2 if the qubit is connected to another repeater register in the opposite bank.
		self.status1 = [[0, 0] for _ in range(len(conn1))]
		self.status2 = [[0, 0] for _ in range(len(conn2))]

		# Keep a first-in-first-out queue of registers to be internally BK-BSM'ed.
		# queue.append() or queue.extend() adds to the right;
		# queue.popleft() pops from the left.
		self.queue1 = deque()
		self.queue2 = deque()

		# Keep track of connections between the banks of repeater registers.
		# 	self.BK_BSM_wait1.get(index) = index of register in the second bank that the register in the
		#								   first bank is connected to
		# 	vice versa for self.BK_BSM_wait2.get(index)
		self.BK_BSM_wait1 = dict()
		self.BK_BSM_wait2 = dict()

		# ClassicalConnection to repeater beamsplitter.
		self.repeater_bs_conn = None

		self.to_print = to_print

	def set_repeater_bs_conn(self, repeater_bs_conn):
		self.repeater_bs_conn = repeater_bs_conn
	
	def set_planner_conn(self, planner_conn):
		self.planner_conn = planner_conn

	def from_planner_conn(self, args = None):
		# Messages sent from central planner.
		info, _ = self.planner_conn.get_as(self.myID)
		for data in info:
			if isinstance(data, tuple) and data[0] == "clock_cycle":
				self.clock_cycle()
			elif isinstance(data, tuple) and data[0] == "do_BSM":
				self.do_BSM(data)
			elif isinstance(data, tuple) and data[0] == "unitary":
				self.apply_unitary(data)

	def clock_cycle(self):
		# Called at the start of each network clock cycle by the central planner.
		# Unoccupied registers in the hybrid repeater should try Barrett-Kok with their counterparts.
		# Send start signals to all registers to start BK.
		# Should always start on the connection with adjacent nodes, since BK-BSM attempts occur
		# in one clock cycle in the hybrid repeater.
		for i in range(len(self.status1)):
			if self.status1[i][0] == 0 and self.status1[i][1] != 1:
				# The nuclear spin should not already be entangled to the adjacent node.
				# If the nuclear spin is already entangled to the adjacent node, then the
				# electron spin should not try to entangle itself to the adjacent node as well.
				self.conn1[i].put_from(self.myID, data = [('start', 1)])
		for i in range(len(self.status2)):
			if self.status2[i][0] == 0 and self.status2[i][1] != 1:
				self.conn2[i].put_from(self.myID, data = [('start', 1)])

	def do_BSM(self, data):
		_, index, other_index = data
		self.conn1[index].put_from(self.myID, data = [('BSM',)])
		self.conn2[other_index].put_from(self.myID, data = [('BSM',)])

	def apply_unitary(self, data):
		# Method to run when the central planner wants to apply a unitary based on an adjacent BSM.
		_, bank, index, isX, isZ = data
		if bank == 1:
			# Identify whether the qubit that was connected to an adjacent node is in the electron
			# or nuclear spin.
			isnuclear = (self.status1[index][1] == 1)
			assert isnuclear or (self.status1[index][0] == 1), str(self.status1[index])
			if isX == 1: nq.operate(self.rep1[index].qmemory.get_qubit(isnuclear), ns.X)
			if isZ == 1: nq.operate(self.rep1[index].qmemory.get_qubit(isnuclear), ns.Z)
		elif bank == 2:
			isnuclear = (self.status2[index][1] == 1)
			assert isnuclear or (self.status2[index][0] == 1), str(self.status2[index])
			if isX == 1: nq.operate(self.rep2[index].qmemory.get_qubit(isnuclear), ns.X)
			if isZ == 1: nq.operate(self.rep2[index].qmemory.get_qubit(isnuclear), ns.Z)
	
	def is_entangled(self, bank, index, args = None):
		# Method to run when the repeater registers inform the control that entanglement has been
		# established with adjacent nodes.
		# The side can be determined by setting up the callback-channel linkage appropriately.
		if bank == 1:
			[data,], _ = self.conn1[index].get_as(self.myID)
			if self.to_print: print(self.myID, bank, index, data, self.queue1, self.queue2)
			# Entanglement with some qubit.
			if isinstance(data, tuple) and data[0] == "success":
				self.status1[index][0] = data[1] # the BK side corresponds to the status.
				# Entanglement with adjacent node.
				if data[1] == 1:
					self.queue1.append(index)
					# Move entangled qubit to the nuclear spin.
					self.conn1[index].put_from(self.myID, data = [('move_qubit',)])
				# Success of local Barrett-Kok.
				elif data[1] == 2:
					# The change in status indicates that the local BK has succeeded.
					# The companion register (in the opposite bank) should check that the status of
					# this register is [2, 1].
					# Then the 2BSM can be performed.
					other_index = self.BK_BSM_wait1[index]
					# If the companion node is also done, do the 2BSM.
					# However, we need to apply the appropriate unitaries to the adjacent nodes, so
					# we don't want to do all the BSM simultaneously.
					# Inform the central planner, who will tell the repeater control when to do the BSM.
					if self.status2[other_index][0] == 2:
						if self.to_print: print(self.status1[index], self.status2[other_index])
						assert self.status1[index][1] == 1
						assert self.status2[other_index][1] == 1
						self.planner_conn.put_from(self.myID, \
														data = [('ready_BSM', index, other_index)])
						self.repeater_bs_conn.put_from(self.myID, \
														data = [('remove', index, other_index),])
						if self.to_print: print(self.myID, bank, index, 'remove', index, other_index)
					else: assert self.status2[other_index][0] == 0
			# Success of local Bell state measurements.
			elif isinstance(data, tuple) and data[0] == "BSM_success":
				# We should record the pairs of register indices that were connected by BK+2BSM.
				# A successful BSM is indicated by popping from self.BK_BS_wait1.
				other_index = self.BK_BSM_wait1[index]
				self.BK_BSM_wait1.pop(index)
				# Inform the central planner so that it can perform the appropriate unitaries on
				# the adjacent nodes.
				# The central planner needs to know both BSM results, it would also be helpful to know
				# the side on which the BSM was performed.
				# Idea: (0, 0) = (01+10) is what we want, so no flip required
				# 		(0, 1) = (01-10) requires a parity flip (Z)
				#		(1, 0) = (00+11) requires a population flip (X)
				#		(1, 1) = (00-11) requires a population + parity flip (Z, X order doesn't matter
				#												 			  up to a phase)
				# Specifically for Bell states,
				# 	2 population flips (one on each side) cancel out, and
				# 	2 parity flips (one on each side) cancel out.
				results = data[2]
				self.status1[index][0], self.status1[index][1] = 0, 0
				# Message format: (index of register in bank 1, index of register in bank 2,
				#					bank: 1 or 2, BSM results: (0,0), (0,1), (1,0), (1,1))
				self.planner_conn.put_from(self.myID, data = [(index, other_index, bank, results),])
			# Success of local SWAP gate.
			elif data[0] == "move_qubit_success":
				self.status1[index][1] = self.status1[index][0]
				self.status1[index][0] = 0
				# After moving the qubit (see while loop below), run BK+2BSM repeatedly.
		elif bank == 2:
			[data,], _ = self.conn2[index].get_as(self.myID)
			if self.to_print: print(self.myID, bank, index, data, self.queue1, self.queue2)
			if isinstance(data, tuple) and data[0] == "success":
				self.status2[index][0] = data[1]
				if data[1] == 1:
					self.queue2.append(index)
					self.conn2[index].put_from(self.myID, data = [('move_qubit',)])
				elif data[1] == 2:
					other_index = self.BK_BSM_wait2[index]
					if self.status1[other_index][0] == 2:
						if self.to_print: print(self.status1[other_index], self.status2[index], \
										self.BK_BSM_wait1[other_index], self.BK_BSM_wait2[index])
						assert self.status1[other_index][1] == 1
						assert self.status2[index][1] == 1
						self.planner_conn.put_from(self.myID, \
														data = [('ready_BSM', other_index, index),])
						self.repeater_bs_conn.put_from(self.myID, \
														data = [('remove', other_index, index),])
						if self.to_print: print(self.myID, bank, index, 'remove', other_index, index)
					else: assert self.status1[other_index][0] == 0
			elif isinstance(data, tuple) and data[0] == "BSM_success":
				other_index = self.BK_BSM_wait2[index]
				self.BK_BSM_wait2.pop(index)
				results = data[2]
				self.status2[index][0], self.status2[index][1] = 0, 0
				self.planner_conn.put_from(self.myID, data = [(other_index, index, bank, results),])
			elif data[0] == "move_qubit_success":
				self.status2[index][1] = self.status2[index][0]
				self.status2[index][0] = 0
				# After moving the qubit (see while loop below), run BK+2BSM repeatedly.

		# If both banks have at least one successful entanglement on each side, we can perform the
		# required BK+2BSM.
		while len(self.queue1) * len(self.queue2) > 0:
			counter1 = self.queue1.popleft()
			counter2 = self.queue2.popleft()
			self.BK_BSM_wait1[counter1] = counter2
			self.BK_BSM_wait2[counter2] = counter1
			# Perform an effective Bell state measurement (i.e. BK+2BSM) using the following process:
			#	0. [already done] move repeater qubit from the electron spin to the nuclear spin
			# 	1. establish entanglement between electron spins
			#	2. perform two BSMs at both registers
			# These will be done at the repeater nodes directly.
			# Note that all repeater registers are connected to a single repeater beamsplitter.
			# The repeater control will tell the repeater beamsplitter which nodes should be combined
			# appropriately.
			self.repeater_bs_conn.put_from(self.myID, data = [('add', counter1, counter2),])
			if self.to_print: print(self.myID, bank, index, 'add', counter1, counter2)

	def run_protocol(self):
		# Notify repeater nodes that need to start local Barrett-Kok.
		for index1 in range(len(self.status1)):
			# The repeater register doesn't need to know which register in the other bank it will be
			# performing the BK+2BSM with, because that is handled by the repeater control and the
			# beamsplitter.
			# We just need to verify that the register is indeed lined up for BK+2BSM.
			# Otherwise there is no need for local entanglement yet.
			if self.status1[index1][0] == 0 and self.BK_BSM_wait1.get(index1) is not None:
				if self.to_print: print(self.myID, 1, index1, 'start', 2)
				self.conn1[index1].put_from(self.myID, data = [('start', 2)]) # local conn
		for index2 in range(len(self.status2)):
			if self.status2[index2][0] == 0 and self.BK_BSM_wait2.get(index2) is not None:
				if self.to_print: print(self.myID, 2, index2, 'start', 2)
				self.conn2[index2].put_from(self.myID, data = [('start', 2)])
			
class PlannerControlProtocol(TimedProtocol):
	''' Protocol for central planner. Assume a linear repeater chain with a left and right.'''
	def __init__(self, time_step, node, left_scont, right_scont, left_conn, right_conn, rep_conns,\
						to_print = True):
		# time_step = network clock cycle
		# rep_conns = ClassicalConnections to repeater controls
		super().__init__(time_step, node = node, connection = None)
		self.myID = self.node.nodeID

		self.left_scont = left_scont # really want ScontProtocol, so we can extract the registers
		self.right_scont = right_scont
		self.left_conn = left_conn
		self.right_conn = right_conn
		self.rep_conns = rep_conns
		self.num_rep = len(rep_conns)

		# self.left2right[register] = rightmost entangled register
		# registers are specified by (index in rep_conns, index in bank) or
		#	('left', original index in bank, index among all historical qubits in the leftmost bank) or
		#	('right', ...)
		# (Indices start from 0.)
		self.left2right = dict()
		self.right2left = dict()

		# Stores left and right qubits.
		self.left_qubits = []
		self.right_qubits = []

		# Data collection.
		# self.key_times[i] = sim_time at the (i+1)-th time the two ends share an entangled qubit.
		self.key_times = []
		# self.key_dm[i] = density matrix of the (i+1)-th time the two ends share an entangled qubit.
		self.key_dm = []
		# All data.
		self.data = [self.key_times, self.key_dm]

		# Waiting for two BSM results to come back from the same repeater node.
		self.waiting = dict()
		# Whether the chain is waiting for 2BSM+unitaries to be done.
		self.is_holding = False
		self.holding_BSM = deque()

		self.to_print = to_print

	def local_link(self, node, args):
		# node = integer representing an index in rep_conns
		# Two types of information are passed to the central planner: when the registers are ready
		# to do the BSM, and when BSM results are obtained.
		info, _ = self.rep_conns[node].get_as(self.myID)
		for data in info:
			if data[0] == 'ready_BSM': # indicator of readiness to do BSM
				if not self.is_holding: 
					self.rep_conns[node].put_from(self.myID, data = [('do_BSM',) + data[1:]])
					self.is_holding = True
				else:
					self.holding_BSM.append((node, data[1:]))
			else: # BSM results
				index1, index2, bank, results = data
				self.is_holding = False
				if self.to_print: print(node, data)
				if self.waiting.get((node, index1, index2)) is None:
					# The new connection is between bank 2 of self.right2left[node] and bank 1 of 
					# self.left2right[node]; note that if local_link is called but 
					# self.right2left[node] is None, then there is an implicit connection with 
					# node-1 on the left. (Conversely on the right.)
					lbell = self.right2left.get((node, index1))
					# Clear the connections so they can be updated later.
					if lbell is not None: self.right2left.pop((node, index1))
					# Handle the case where the connection is implicit.
					if lbell is None and node != 0: lbell = (node - 1, index1)
					elif lbell is None and node == 0: 
						lbell = ('left', index1, len(self.left_qubits))
						self.left_qubits.append(self.left_scont.sources[index1].qmem.get_qubit(0))
						self.left_conn.put_from(self.myID, data = [('free', index1),])
						# If infinite bank, the left node can free up index1 for other 
						# entanglement attempts.
						# If finite bank, do not free up index1 yet.
					rbell = self.left2right.get((node, index2))
					if rbell is not None: self.left2right.pop((node, index2))
					if rbell is None and node != self.num_rep-1: rbell = (node + 1, index2)
					elif rbell is None and node == self.num_rep-1: 
						rbell = ('right', index2, len(self.right_qubits))
						self.right_qubits.append(self.right_scont.sources[index2].qmem.get_qubit(0))
						self.right_conn.put_from(self.myID, data = [('free', index2),])
					# Update the connections.
					self.right2left[rbell] = lbell # originally (node, index2)
					self.left2right[lbell] = rbell # originally (node, index1)
					self.waiting[(node, index1, index2)] = (lbell, rbell)
					is_complete = False
				else: # (node, index1, index2) appears twice, once for each BSM
					lbell, rbell = self.waiting.pop((node, index1, index2))
					# Check whether this connects the two ends of the repeater chain.
					is_complete = (lbell[0] == 'left') and (rbell[0] == 'right')
				
				# Sanity check.
				assert lbell[0] != 'right'
				assert rbell[0] != 'left'

				# Apply unitary for BSM.
				if bank == 1:
					if lbell[0] == 'left':
						lqubit = self.left_qubits[lbell[2]] # qubits from the list
															# (works for both infinite and finite banks)
						if results[0] == 1: nq.operate(lqubit, ns.X)
						if results[1] == 1: nq.operate(lqubit, ns.Z)
					else:
						self.rep_conns[lbell[0]].put_from(self.myID, \
												data=[("unitary", 2, lbell[1], results[0], results[1]),])
				elif bank == 2:
					if rbell[0] == 'right':
						rqubit = self.right_qubits[rbell[2]]
						if results[0] == 1: nq.operate(rqubit, ns.X)
						if results[1] == 1: nq.operate(rqubit, ns.Z)
					else:
						self.rep_conns[rbell[0]].put_from(self.myID, \
												data=[("unitary", 1, rbell[1], results[0], results[1]),])
					
				# Check if any entanglement has been completed.
				if is_complete:
					if self.to_print: print(node, 'complete!', lbell[1], rbell[1])
					# Extract data.
					self.key_times.append(sim_time())
					self.key_dm.append(nq.reduced_dm([self.left_qubits[lbell[2]], \
																self.right_qubits[rbell[2]]]))
					if self.to_print: print(sim_time(), '\n', self.key_dm[-1])
					# Inform nodes.
					self.left_conn.put_from(self.myID, data = [('complete', lbell[1]),])
					self.right_conn.put_from(self.myID, data = [('complete', rbell[1]),])
					# If infinite bank, these would already have been freed up for other entanglements.
					# If finite bank, free up these registers for further entanglements.

				# Do next BSM, if any.
				if len(self.holding_BSM) > 0:
					nn, dd = self.holding_BSM.popleft() # node, data
					self.rep_conns[nn].put_from(self.myID, data = [('do_BSM',) + dd])
					self.is_holding = True

	def run_protocol(self):
		# Set network clock cycle.
		# Notify all nodes that they can start Barrett-Kok over long distances.
		if self.to_print: print('clock cycle', sim_time(), self.left2right)
		self.left_conn.put_from(self.myID, data = [("clock_cycle",),])
		self.right_conn.put_from(self.myID, data = [("clock_cycle",),])
		for conn in self.rep_conns:
			conn.put_from(self.myID, data = [("clock_cycle",),])

class ScontProtocol(TimedProtocol):
	''' Protocol for source control.'''
	def __init__(self, time_step, node, sources, conns, is_infinite_bank = False, to_print = False):
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.sources = sources # source nodes
		self.conns = conns

		# Connection to central planner.
		self.planner_conn = None

		# Status = 1 if the source node is entangled with the adjacent repeater node.
		self.status = [0]*len(conns)

		self.to_print = to_print

		# Whether we treat the ends of the repeater chain have infinite banks of registers.
		self.is_infinite_bank = is_infinite_bank

		self.to_print = to_print
	
	def set_planner_conn(self, planner_conn):
		self.planner_conn = planner_conn
	
	# The source control interface with the planner control by:
	# 	1. running Barrett-Kok at every network clock cycle;
	#	2. keeping track of which registers are available for Barrett-Kok.
	# Note that the central planner does the appropriate unitaries to correct for BSM results.
	# We also assume that the time needed to communicate heralded entanglement across the chain is
	# much shorter than the time needed to establish that entanglement in the first place
	# (valid if the success probability of entanglement across a single link is small).
	
	def from_planner_conn(self, args = None):
		info, _ = self.planner_conn.get_as(self.myID)
		for data in info:
			if isinstance(data, tuple) and data[0] == "clock_cycle":
				self.clock_cycle()
			elif isinstance(data, tuple) and data[0] == "free":
				# The register is entangled past the first repeater node.
				if self.is_infinite_bank: self.free_up(data)
			elif isinstance(data, tuple) and data[0] == "complete":
				# The register has successfully been entangled across the whole chain.
				if not self.is_infinite_bank: self.free_up(data)
	
	def clock_cycle(self):
		# Called at the start of each network clock cycle by the central planner.
		# Unoccupied registers in the hybrid repeater should try Barrett-Kok with their counterparts.
		# Send start signals to all registers to start BK.
		# Should always start on the connection with adjacent nodes, since BK-BSM attempts occur
		# in one clock cycle in the hybrid repeater.
		for i in range(len(self.status)):
			if self.status[i] == 0:
				self.conns[i].put_from(self.myID, data = ['start'])

	def free_up(self, data):
		# Sanity check.
		if self.to_print: print(self.myID, 'free_up', data)
		assert self.status[data[1]] != 0
		# Free up a register to do future Barrett-Kok.
		self.status[data[1]] = 0

	def is_entangled(self, index, args = None):
		# Method to run when the source registers inform the scont that entanglement is successful.
		[data,], _ = self.conns[index].get_as(self.myID)
		# Note that source registers send and receive classical messages as single strings.
		if data == 'success':
			self.status[index] = 1 # hold this register until entanglement has been achieved across
								   # the whole board

class RepeaterBSProtocol(TimedProtocol):
	''' 
	Protocol for repeater beam splitter, which should always be paired with a repeater control node. 
	Main purpose: reroute connections between companion repeater registers to the same beamsplitter/
	detector pairs.
	'''
	def __init__(self, time_step, node, conn_to_rep_control, conn1, conn2, test_conn1, test_conn2):
		super().__init__(time_step, node = node, connection = None)
		self.myID = self.node.nodeID

		# Connection to the repeater control.
		self.conn_to_rep_control = conn_to_rep_control
		# May not be used for outgoing messages?
		# QuantumConnection to/from the repeater registers.
		self.conn1 = conn1
		self.conn2 = conn2
		# ClassicalConnection to/from the repeater registers.
		# e.g. test_conn1[i] refers to a connection from index1 = i
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

	def set_repconn(self, repconn1, repconn2, test_repconn1, test_repconn2):
		# repconn1[i][j] = connection from the repeater BS to the constituent BS used when
		#					index1 = i and index2 = j are trying to connect, for qubits that come
		#					from the first bank
		# repconn2[i][j] = same as above, but for qubits that come from the second bank
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
	
	def from_rep_registers(self, side, index, args = None):
		# Route data from the repeater registers to the appropriate beamsplitters.
		if side == 1:
			data, _ = self.conn1[index].get_as(self.myID)
			# Find out who to route it to.
			other_index = self.from1to2[index]
			self.repconn1[index][other_index].put_from(self.myID, data = data)
		elif side == 2:
			data, _ = self.conn2[index].get_as(self.myID)
			other_index = self.from2to1[index]
			self.repconn2[other_index][index].put_from(self.myID, data = data)

	def from_bs(self, side, index, args = None):
		# Route data from the beamsplitters to the appropriate repeater registers.
		if side == 1:
			[msg,], _ = self.test_repconn1[index][self.from1to2[index]].get_as(self.myID)
			self.test_conn1[index].put_from(self.myID, data = [msg,])
			#print('repBS', self.test_conn1[index].idA, self.test_conn1[index].idB)
		elif side == 2:
			[msg,], _ = self.test_repconn2[self.from2to1[index]][index].get_as(self.myID)
			self.test_conn2[index].put_from(self.myID, data = [msg,])
			#print('repBS', self.test_conn2[index].idA, self.test_conn2[index].idB)
		# Note that the repeater registers will inform the repeater control of success/failure directly.

class LongDistanceLink:
	''' 
	Registers, connections, protocols associated with a long-distance link between adjacent
	repeater nodes.
	We focus on links as the primary object in a repeater chain. Each link has multiple modes,
	and each mode has an associated entanglement protocol (in this case, Barrett-Kok).
	By construction, there should be as many registers on each side of the link as modes.
	The network has an implicit left-to-right directionality.

	Convention: the register on the right side of the first Barrett-Kok link will do the 
				BK phase correction.
	'''
	def __init__(self, index_gen, num_modes, left_correct = False, right_correct = True):
		# index_gen = (global) index generator
		# num_modes = degree of parallelism
		self.index_gen = index_gen
		self.num_modes = num_modes
		# Whether to correct for BK phase.
		self.left_correct = left_correct
		self.right_correct = right_correct

		### Relevant link parameters.
		# Clock rates.
		self.BASE_CLOCK_RATE = 10 # for nodes that don't rely on frequent checks, 
								  # especially reactive protocols
		self.BS_CLOCK_RATE = None # for BeamSplitterProtocol and StateCheckProtocol; defines
								  # whether two photons arrived simultaneously
		# Coherence times:
		#	a list of [T1, T2] the length of which is equal to the number of memories
		self.left_times = None
		self.right_times = None
		# Quantum losses and time delays across a link.
		self.channel_loss = None
		self.link_delay = None
		# Interaction noise on nuclear spin.
		self.noise_on_nuclear = None
		# Fidelities.
		self.gate_fidelity = None
		self.meas_fidelity = None
		self.prep_fidelity = None
		self.detector_pdark = None
		self.detector_eff = None
		# Barrett-Kok.
		self.reset_delay = None
		# Have params been set?
		self.is_params = False
	
		### Relevant objects.
		# Nodes on either end of the link.
		self.left_nodes = []
		self.right_nodes = []
		# Protocols for these nodes.
		self.left_prots = []
		self.right_prots = []

	def set_params(self, BS_CLOCK_RATE, left_times, right_times, channel_loss, link_delay,\
					noise_on_nuclear, gate_fidelity, meas_fidelity, prep_fidelity, \
					detector_pdark, detector_eff, reset_delay):
		self.BS_CLOCK_RATE = BS_CLOCK_RATE
		self.left_times = left_times
		self.right_times = right_times
		self.channel_loss = channel_loss
		self.link_delay = link_delay
		self.noise_on_nuclear = noise_on_nuclear
		self.gate_fidelity = gate_fidelity
		self.meas_fidelity = meas_fidelity
		self.prep_fidelity = prep_fidelity
		self.detector_pdark = detector_pdark
		self.detector_eff = detector_eff
		self.reset_delay = reset_delay
		self.is_params = True

		# Make the relevant functions.
		# x = NetSquid node name
		# make_left_memory, make_right_memory = functions that take a NetSquid node name,
		#										and return a MemoryDevice (could be noisy)
		#	Example:
		#		make_memory = lambda x: UniformStandardMemoryDevice(x, 2, T2 = 0.01)
		#	The specific device depends on the types of network nodes present on either side of the
		#	link. Note that there is an implicit "left" and "right" in the repeater chain.
		self.make_left_memory = lambda x: StandardMemoryDevice(x, len(left_times), \
																		decoherenceTimes = left_times)
		self.make_right_memory = lambda x: StandardMemoryDevice(x, len(right_times), \
																		decoherenceTimes = right_times)
		# x, y = nodes to be connected (e.g. x = repeater register, y = beamsplitter)
		self.node_to_bs = lambda x, y: QuantumConnection(x, y, \
												noise_model = QubitLossNoiseModel(channel_loss),\
												delay_model = FixedDelayModel(delay=link_delay))
		self.bs_to_det = lambda x, y: QuantumConnection(x, y)
		# x, y, z = node, QuantumConnection, ClassicalConnection
		self.make_left_prot = lambda x, y, z, to_correct = self.left_correct: \
								SourceProtocol(self.BASE_CLOCK_RATE, x, y, z, to_correct=to_correct,\
											   prep_fidelity = prep_fidelity, \
											   reset_delay = reset_delay) if len(left_times) == 1\
								else BellProtocol(self.BASE_CLOCK_RATE, x, y, z, to_correct=to_correct,\
												  noise_on_nuclear = noise_on_nuclear, \
												  gate_fidelity = gate_fidelity,
											 	  meas_fidelity = meas_fidelity,\
												  prep_fidelity = prep_fidelity, \
												  reset_delay = reset_delay)
		self.make_right_prot = lambda x, y, z, to_correct = self.right_correct: \
								SourceProtocol(self.BASE_CLOCK_RATE, x, y, z, to_correct=to_correct,\
											   prep_fidelity = prep_fidelity, \
											   reset_delay = reset_delay) if len(right_times) == 1\
								else BellProtocol(self.BASE_CLOCK_RATE, x, y, z, to_correct=to_correct,\
												  noise_on_nuclear = noise_on_nuclear, \
												  gate_fidelity = gate_fidelity,
											 	  meas_fidelity = meas_fidelity,\
												  prep_fidelity = prep_fidelity,\
												  reset_delay = reset_delay)
		self.make_det_prot = lambda x, y, z: DetectorProtocol(self.BASE_CLOCK_RATE, x, y, z, \
												pdark = detector_pdark, efficiency = detector_eff)
		# x, y, z = node, inbound, outbound
		self.make_bs_prot = [lambda x, y, z: BeamSplitterProtocol(BS_CLOCK_RATE, x, y, z),\
							 lambda x, y, z: StateCheckProtocol(BS_CLOCK_RATE, x, y, z)]
		return self.make_BK

	def make_BK(self):
		if not self.is_params: raise Exception("link parameters need to be defined")
		return self.create_BK(self.index_gen, self.make_left_memory, self.make_right_memory,\
							  self.node_to_bs, self.bs_to_det,\
							  self.make_left_prot, self.make_right_prot,\
							  self.make_det_prot, self.make_bs_prot,\
							  self.link_delay)

	def make_link(self):
		for i in range(self.num_modes):
			[a, b], [c, d] = self.make_BK()
			self.left_nodes.append(a)
			self.right_nodes.append(b)
			self.left_prots.append(c)
			self.right_prots.append(d)

	def get_nodes(self, side):
		if side == 1:
			return self.left_nodes, self.left_prots
		elif side == 2:
			return self.right_nodes, self.right_prots

	# Outline for creating a Barrett-Kok link (with an associated mode).
	@staticmethod
	def create_BK(index_gen, make_left_memory, make_right_memory, node_to_bs, bs_to_det, \
					make_left_prot, make_right_prot, make_det_prot, make_bs_prot, \
					classical_delay = 0.):
		# Inputs:
		#	 index_gen = generator for node indices
		#	 make_left_memory, make_right_memory = functions that return MemoryDevice objects
		#	 node_to_bs = function that takes the sender node and the beamsplitter node, and
		#				   returns a QuantumConnection between the two
		#	 	Example: 
		#			node_to_bs = lambda x, y: QuantumConnection(x, y, \
		#											noise_model = QubitLossNoiseModel(0.1))
		#	 bs_to_det = function that takes the beamsplitter node and the detector node, and
		#				  returns a QuantumConnection between the two
		#				  Note: dark counts can be implemented here, with reverse amplitude damping.
		#	 make_left_prot, make_right_prot, make_det_prot = 
		#				  functions that take (node, QuantumConnection, ClassicalConnection) and
		#				  returns the appropriate protocol
		#		Example:
		#		make_right_prot = lambda x, y, z: BellProtocol(x, y, z)
		#		Note on BellProtocol: we only add the repeater connection later.
		#	 make_bs_prot = [BeamSplitterProtocol.__init__, StateCheckProtocol.__init__]
		#	 classical_delay = time delay (in ns) for connection between nodes and beamsplitters
		# Returns:
		#	[left_node, right_node], [left_prot, right_prot]

		index = index_gen.__next__()
		left_atom = make_left_memory("atom"+str(index))
		left_node = QuantumNode("source"+str(index), index, memDevice = left_atom)
		index = index_gen.__next__()
		right_atom = make_right_memory("atom"+str(index))
		right_node = QuantumNode("source"+str(index), index, memDevice = right_atom)
		# Detectors.
		index = index_gen.__next__()
		detector1 = QuantumNode("detector"+str(index), index)
		index = index_gen.__next__()
		detector2 = QuantumNode("detector"+str(index), index)
		# Beamsplitter.
		index = index_gen.__next__()
		beamsplitter = QuantumNode("beamsplitter"+str(index), index)
		# Quantum connections from node to beamsplitter.
		conn1 = node_to_bs(left_node, beamsplitter)
		conn2 = node_to_bs(right_node, beamsplitter)
		# Quantum connections from beamsplitter to detectors.
		conn3 = bs_to_det(beamsplitter, detector1)
		conn4 = bs_to_det(beamsplitter, detector2)
		# Classical connections from detectors to beamsplitter to nodes.
		test_conn1 = ClassicalConnection(detector1, beamsplitter)
		test_conn2 = ClassicalConnection(detector2, beamsplitter)
		test_conn3 = ClassicalConnection(beamsplitter, left_node, \
											delay_model = FixedDelayModel(delay=classical_delay))
		test_conn4 = ClassicalConnection(beamsplitter, right_node, \
											delay_model = FixedDelayModel(delay=classical_delay))
		# Set up protocols at nodes, detectors and beamsplitter.
		# Note that the atom/repeater protocol must have a "verification" method as the callback.
		proto1 = make_left_prot(left_node, conn1, test_conn3)
		proto2 = make_right_prot(right_node, conn2, test_conn4)
		proto3 = make_det_prot(detector1, conn3, test_conn1)
		proto4 = make_det_prot(detector2, conn4, test_conn2)
		BSproto = make_bs_prot[0](beamsplitter, [conn1, conn2], [conn3, conn4])
		SCproto = make_bs_prot[1](beamsplitter, [test_conn1, test_conn2], \
									[test_conn3, test_conn4])
		# Set up handlers.
		# setup_connection does things automatically, but we can also register handlers manually
		# especially for multi-connection nodes.
		left_node.setup_connection(conn1, [proto1])
		right_node.setup_connection(conn2, [proto2])
		func3 = lambda args, side = 1: proto1.verification(side, args)
		func4 = lambda args, side = 1: proto2.verification(side, args)
		test_conn3.register_handler(left_node.nodeID, func3)
		test_conn4.register_handler(right_node.nodeID, func4)
		conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
		conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
		test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
		test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)
		detector1.setup_connection(conn3, [proto3])
		detector2.setup_connection(conn4, [proto4])
		# Note that the repeater protocol must also be added as a node protocol for a
		# repeater node: see run_protocol in e.g. BellProtocol.
		# This does nothing if run_protocol does nothing.
		left_node.add_node_protocol(proto1)
		left_node.start()
		right_node.add_node_protocol(proto2)
		right_node.start()
		# Return nodes and protocols.
		# Keep the beamsplitter + detectors hidden away in the abstraction.
		return [left_node, right_node], [proto1, proto2]
			
class HybridRepeater:
	''' Registers, connections and protocols associated with a router / hybrid repeater. '''
	def __init__(self, index_gen, num_channels):
		# index_gen = (global) index generator
		# num_channels = degree of parallelism (m/2 for hybrid, m for traditional, 
		#										where m = number of total registers)
		self.index_gen = index_gen
		if isinstance(num_channels, int):
			self.num_channels = (num_channels, num_channels)
		else: self.num_channels = num_channels

		### NetSquid nodes
		# Banks of registers.
		self.reg1 = None
		self.reg2 = None
		# Repeater control.
		self.rep_control = None
		# Repeater beamsplitter.
		self.repBS = None
		# We don't need to hold on to the constituent beamsplitter nodes.

		### NetSquid protocols
		# Protocols for registers.
		self.proto1 = None
		self.proto2 = None
		# Protocol for repeater control.
		self.rep_control_prot = None
		# Protocol for repeater beamsplitter.
		self.repBS_prot = None

		### Parameters
		# As usual, note that rates are actually time intervals.
		self.BASE_CLOCK_RATE = 10
		self.BS_CLOCK_RATE = None
		self.REPEATER_CONTROL_RATE = None # how frequently local BK+2BSM should be tried
		self.repeater_channel_loss = None
		self.local_delay = None
		self.detector_pdark = None
		self.detector_eff = None
	
	def set_registers(self, side, reg, prots):
		assert len(reg) == self.num_channels[side-1]
		assert len(prots) == len(reg)
		if side == 1:
			self.reg1, self.proto1 = reg, prots
		elif side == 2:
			self.reg2, self.proto2 = reg, prots

	def set_params(self, BS_CLOCK_RATE, REPEATER_CONTROL_RATE, repeater_channel_loss, local_delay,\
					detector_pdark, detector_eff):
		self.BS_CLOCK_RATE = BS_CLOCK_RATE
		self.REPEATER_CONTROL_RATE = REPEATER_CONTROL_RATE
		self.repeater_channel_loss = repeater_channel_loss
		self.local_delay = local_delay
		self.detector_pdark = detector_pdark
		self.detector_eff = detector_eff

		# x, y, z, a, b = node, reg1, conn1, reg2, conn2
		self.make_control_prot = lambda x, y, z, a, b: RepeaterControlProtocol(REPEATER_CONTROL_RATE,\
																				x, y, z, a, b)
		self.node_to_bs_rep = lambda x, y: QuantumConnection(x, y, \
											noise_model = QubitLossNoiseModel(repeater_channel_loss),\
											delay_model = FixedDelayModel(delay=local_delay))
		self.bs_to_bs_rep = lambda x, y: QuantumConnection(x, y)
		self.bs_to_det_rep = lambda x, y: QuantumConnection(x, y)
		self.make_rep_bs_prot = lambda x, y, a, b, c, d: \
											RepeaterBSProtocol(self.BASE_CLOCK_RATE, x, y, a, b, c, d)
								
		self.make_bs_prot = [lambda x, y, z: BeamSplitterProtocol(BS_CLOCK_RATE, x, y, z),\
							 lambda x, y, z: StateCheckProtocol(BS_CLOCK_RATE, x, y, z)]
		self.make_det_prot = lambda x, y, z: DetectorProtocol(self.BASE_CLOCK_RATE, x, y, z, \
												to_print = False, \
												pdark = detector_pdark, efficiency = detector_eff)

	def make_rep_control(self):
		assert self.rep_control is None
		next_index = self.index_gen.__next__()
		self.rep_control = QuantumNode("rep_control"+str(next_index), next_index)

		# Connect the repeater registers to the repeater control.
		# Repeater protocols should have a set_repeater_conn method that takes a ClassicalConnection
		# to rep_control as input.
		# Also, repeater control protocols should have an is_entangled method that is run every time
		# a node is successfully entangled by Barrett-Kok.
		
		# Connections.
		conn1 = []
		for i in range(self.num_channels[0]):
			conn = ClassicalConnection(self.reg1[i], self.rep_control)
			self.proto1[i].set_repeater_conn(conn)
			conn1.append(conn)
		conn2 = []
		for i in range(self.num_channels[1]):
			conn = ClassicalConnection(self.reg2[i], self.rep_control)
			self.proto2[i].set_repeater_conn(conn)
			conn2.append(conn)

		# Set up repeater control protocol.
		control_prot = self.make_control_prot(self.rep_control, self.reg1, conn1, self.reg2, conn2)
		# Set up handlers.
		for i in range(len(conn1)):
			# Note that default arguments should go into the left side of the lambda expression.
			func = lambda args, bank = 1, index = i: control_prot.is_entangled(bank, index, args)
			conn1[i].register_handler(self.reg1[i].nodeID, self.proto1[i].start_BK)
			conn1[i].register_handler(self.rep_control.nodeID, func)
		for i in range(len(conn2)):
			# Note that default arguments should go into the left side of the lambda expression.
			func = lambda args, bank = 2, index = i: control_prot.is_entangled(bank, index, args)
			conn2[i].register_handler(self.reg2[i].nodeID, self.proto2[i].start_BK)
			conn2[i].register_handler(self.rep_control.nodeID, func)
		self.rep_control.add_node_protocol(control_prot)
		self.rep_control_prot = control_prot	

	def make_repBS(self):
		assert self.repBS is None
		next_index = self.index_gen.__next__()
		self.repBS = QuantumNode("repBS"+str(next_index), next_index)

		# TODO: clean up the code
		index_gen = self.index_gen
		rep1 = self.reg1
		proto1 = self.proto1
		rep2 = self.reg2
		proto2 = self.proto2
		rep_control = self.rep_control
		rep_control_prot = self.rep_control_prot
		repeater_bs = self.repBS
		make_rep_bs_prot = self.make_rep_bs_prot
		node_to_bs_rep = self.node_to_bs_rep
		bs_to_bs_rep = self.bs_to_bs_rep
		bs_to_det_rep = self.bs_to_det_rep
		make_det_prot = self.make_det_prot
		make_bs_prot = self.make_bs_prot
		classical_rep_delay = self.local_delay
		
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
			proto1[i].set_other_conn(next_conn, next_test_conn)
			conn1.append(next_conn)
			test_conn1.append(next_test_conn)
			# Handlers to register: 
			#	conn1[i] - qubits going into repeater_bs from repeater nodes
			#	test_conn1[i] - success messages going into rep1[i], handled by BK_BSM_verification
		for i in range(len(rep2)):
			next_conn = node_to_bs_rep(rep2[i], repeater_bs)
			next_test_conn = ClassicalConnection(repeater_bs, rep2[i],\
											delay_model = FixedDelayModel(delay=classical_rep_delay))
			proto2[i].set_other_conn(next_conn, next_test_conn)
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
			func = lambda arg, side=1, index=i: rep_bs_prot.from_rep_registers(side, index, arg)
			conn1[i].register_handler(repeater_bs.nodeID, func)
			# Be careful about lambda functions! The index needs to be brought out!
			func2 = lambda arg, side=2, index = i: proto1[index].verification(side, arg)
			test_conn1[i].register_handler(rep1[i].nodeID, func2)
		for i in range(len(rep2)):
			func = lambda arg, side=2, index=i: rep_bs_prot.from_rep_registers(side, index, arg)
			conn2[i].register_handler(repeater_bs.nodeID, func)
			func2 = lambda arg, side=2, index = i: proto2[index].verification(side, arg)
			test_conn2[i].register_handler(rep2[i].nodeID, func2)

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
		self.repBS_prot = rep_bs_prot

class ClientNode:
	''' Registers, connections and protocols associated with a client node. '''
	def __init__(self, index_gen, num_channels, is_infinite_bank = False):
		# index_gen = (global) index generator
		# num_channels = degree of parallelism (m/2 for hybrid, m for traditional, 
		#										where m = number of total registers)
		# is_infinite_bank = whether the client node has (effectively) an infinite number of registers
		#					 or whether they have a finite bank of num_channel registers
		self.index_gen = index_gen
		self.num_channels = num_channels # int

		### NetSquid nodes
		# Banks of registers.
		self.regs = None
		# Source control.
		self.scont = None

		### NetSquid protocols
		# Protocols for registers.
		self.protos = None
		# Protocol for scont.
		self.scont_prot = None

		### Parameters
		# As usual, note that rates are actually time intervals.
		self.BASE_CLOCK_RATE = 10

		### Functions
		# x, y, z = node, sources, conns
		self.make_scont_prot = lambda x, y, z: ScontProtocol(self.BASE_CLOCK_RATE, x, y, z,\
															 is_infinite_bank = is_infinite_bank)

	def set_registers(self, regs, protos):
		assert len(regs) == self.num_channels
		assert len(protos) == len(regs)
		self.regs, self.protos = regs, protos

	def make_scont(self):
		assert self.scont is None
		next_index = self.index_gen.__next__()
		self.scont = QuantumNode("scont"+str(next_index), next_index)

		# Connect the source registers to the source control ("scont").
		# All source protocols should have a "set_scont_conn" method to set up the connection
		# to the central control.
		conns = []
		for i in range(self.num_channels):
			conn = ClassicalConnection(self.scont, self.regs[i])
			self.protos[i].set_scont_conn(conn)
			conns.append(conn)
			conn.register_handler(self.regs[i].nodeID, self.protos[i].start_BK)

		# Set up scont protocol.
		self.scont_prot = self.make_scont_prot(self.scont, self.regs, conns)
		for i in range(len(conns)):
			func = lambda args, index = i: self.scont_prot.is_entangled(index, args)
			conns[i].register_handler(self.scont.nodeID, func)

		self.scont.add_node_protocol(self.scont_prot)

class Chain:
	''' 
	Contains the full repeater chain.
	Assume that all long-distance links have the same number of modes.
	'''
	def __init__(self, index_gen, num_repeaters, num_modes, is_infinite_bank):
		self.index_gen = index_gen # (global) index generator
		self.num_repeaters = num_repeaters # number of repeater nodes, excluding the client nodes
		self.num_modes = num_modes
		self.is_infinite_bank = is_infinite_bank 

		### Nodes.
		self.left_node = None # ClientNode object
		self.right_node = None
		self.rep_nodes = None # list of HybridRepeater objects

		### Long-distance links.
		self.links = None # list of LongDistanceLink objects

		### Central planner.
		self.planner_control = None # NetSquid node
		self.planner_control_prot = None # PlannerControlProtocol object

		### Relevant parameters.
		# Clock rates.
		# BASE_CLOCK_RATE is hardcoded to be 10.
		self.BS_CLOCK_RATE = None
		self.REPEATER_CONTROL_RATE = None # BK+2BSM
		self.NETWORK_RATE = None # across long-distance links
		# Coherence times.
		self.source_times = None # [[T1, T2],] for client registers
		self.rep_times = None # [[T1, T2], [T1, T2]] for repeater registers (electron and nuclear spins)
		# Quantum losses and time delays across a link.
		self.channel_loss = None
		self.link_delay = None
		# Interaction noise on nuclear spin.
		self.noise_on_nuclear = None
		# Fidelities.
		self.gate_fidelity = None
		self.meas_fidelity = None
		self.prep_fidelity = None
		# Detector properties.
		self.detector_pdark = None
		self.detector_eff = None
		# Repeater local properties.
		self.repeater_channel_loss = None
		self.local_delay = None
		# Atom property.
		self.reset_delay = None

		# Collate parameters.
		self.link_params = None # channel_loss onwards
		self.rep_params = None # repeater_channel_loss onwards
	
	# Setting up the network:
	# 	LongDistanceLink objects define the link-layer connectivity of the network.
	#	LongDistanceLink also generates the required registers. This can be done for hybrid / routered
	#	networks because each node has a separate bank of registers for each edge. These banks are
	# 	connected using a local switch.
	# 	We then identify these registers with the appropriate nodes and set up the protocols.
	#	Then we set up the central planner. 
	#	(Neglect communication times with the planner for simplicity.)

	def set_params(self, BS_CLOCK_RATE, REPEATER_CONTROL_RATE, NETWORK_RATE,\
						source_times, rep_times,\
						channel_loss, link_delay,\
						noise_on_nuclear,\
						gate_fidelity, meas_fidelity, prep_fidelity,\
						detector_pdark, detector_eff,\
						repeater_channel_loss, local_delay, reset_delay):
		self.BS_CLOCK_RATE		= BS_CLOCK_RATE
		self.REPEATER_CONTROL_RATE = REPEATER_CONTROL_RATE
		self.NETWORK_RATE		= NETWORK_RATE		
		self.source_times		= source_times		
		self.rep_times			= rep_times			
		self.channel_loss		= channel_loss		
		self.link_delay			= link_delay			
		self.noise_on_nuclear	= noise_on_nuclear	
		self.gate_fidelity		= gate_fidelity		
		self.meas_fidelity		= meas_fidelity		
		self.prep_fidelity		= prep_fidelity		
		self.detector_pdark		= detector_pdark		
		self.detector_eff		= detector_eff		
		self.repeater_channel_loss	= repeater_channel_loss
		self.local_delay		= local_delay
		self.reset_delay		= reset_delay

		self.link_params = [channel_loss, link_delay, noise_on_nuclear, gate_fidelity, meas_fidelity,\
							prep_fidelity, detector_pdark, detector_eff, reset_delay]
		self.rep_params = [repeater_channel_loss, local_delay, detector_pdark, detector_eff]
							 
	def make_links(self):
		self.links = []
		# Make leftmost link.
		lc, rc = False, True
		left_link = LongDistanceLink(self.index_gen, self.num_modes, lc, rc)
		left_link.set_params(self.BS_CLOCK_RATE, self.source_times, self.rep_times, *self.link_params)
		left_link.make_link()
		self.links.append(left_link)
		# Make repeater-repeater links.
		for i in range(self.num_repeaters - 1):
			lc, rc = lc, rc
			link = LongDistanceLink(self.index_gen, self.num_modes, lc, rc)
			link.set_params(self.BS_CLOCK_RATE, self.rep_times, self.rep_times, *self.link_params)
			link.make_link()
			self.links.append(link)
		# Make rightmost link.
		lc, rc = lc, rc
		right_link = LongDistanceLink(self.index_gen, self.num_modes, lc, rc)
		right_link.set_params(self.BS_CLOCK_RATE, self.rep_times, self.source_times, *self.link_params)
		right_link.make_link()
		self.links.append(right_link)
	
	def make_nodes(self):
		self.left_node = ClientNode(self.index_gen, self.num_modes, self.is_infinite_bank)
		self.left_node.set_registers(*self.links[0].get_nodes(1))
		self.left_node.make_scont()
		
		self.rep_nodes = []
		for i in range(self.num_repeaters):
			node = HybridRepeater(self.index_gen, self.num_modes)
			node.set_registers(1, *self.links[i].get_nodes(2))
			node.set_registers(2, *self.links[i+1].get_nodes(1))
			node.set_params(self.BS_CLOCK_RATE, self.REPEATER_CONTROL_RATE, *self.rep_params)
			node.make_rep_control()
			node.make_repBS()
			self.rep_nodes.append(node)

		self.right_node = ClientNode(self.index_gen, self.num_modes, self.is_infinite_bank)
		self.right_node.set_registers(*self.links[-1].get_nodes(2))
		self.right_node.make_scont()

		text = str(self.left_node.scont.nodeID) + '--'
		for i in range(self.num_repeaters):
			text += str(self.rep_nodes[i].rep_control.nodeID) + '--'
		text += str(self.right_node.scont.nodeID)
		print(text)
		
	def make_planner_control(self):
		next_index = self.index_gen.__next__()
		self.planner_control = QuantumNode("planner_control"+str(next_index), next_index)

		# Set up connections to source/repeater controls.
		left_conn = ClassicalConnection(self.left_node.scont, self.planner_control)
		right_conn = ClassicalConnection(self.right_node.scont, self.planner_control)
		rep_conns = [ClassicalConnection(self.rep_nodes[i].rep_control, self.planner_control) \
																for i in range(self.num_repeaters)]
		self.left_node.scont_prot.set_planner_conn(left_conn)
		self.right_node.scont_prot.set_planner_conn(right_conn)
		for i in range(self.num_repeaters): 
			self.rep_nodes[i].rep_control_prot.set_planner_conn(rep_conns[i])
		# Set up protocol.
		self.planner_control_prot = PlannerControlProtocol(self.NETWORK_RATE, self.planner_control,\
										self.left_node.scont_prot, self.right_node.scont_prot,\
										left_conn, right_conn, rep_conns)
		# Set up handlers.
		left_conn.register_handler(self.left_node.scont.nodeID, \
													self.left_node.scont_prot.from_planner_conn)
		right_conn.register_handler(self.right_node.scont.nodeID, \
													self.right_node.scont_prot.from_planner_conn)
		# The leftmost and rightmost nodes don't talk to the planner control.
		for i in range(self.num_repeaters):
			rep_conns[i].register_handler(self.rep_nodes[i].rep_control.nodeID,\
												self.rep_nodes[i].rep_control_prot.from_planner_conn)
			func = lambda args, node = i: self.planner_control_prot.local_link(node, args)
			rep_conns[i].register_handler(self.planner_control.nodeID, func)
		self.planner_control.add_node_protocol(self.planner_control_prot) 
		# Necessary for self.planner_control.start() to activate run_protocol().

	def start(self):
		self.planner_control.start()
		for i in range(self.num_repeaters):
			self.rep_nodes[i].rep_control.start()

def run_simulation(num_repeaters, num_modes, source_times, rep_times, channel_loss, duration = 100, \
					repeater_channel_loss = 0., noise_on_nuclear_params = None, \
					link_delay = 0., link_time = 1., local_delay = 0., local_time = 0.2, \
					time_bin = 0.01, detector_pdark = 1e-7, detector_eff = 0.93,\
					gate_fidelity = 0.999, meas_fidelity = 0.9998, prep_fidelity = 0.99, \
					reset_delay = 0.1):
	## Parameters:
	# num_repeaters = number of repeaters in the hybrid chain
	# num_modes = degree of parallelism (m/2 for hybrid, m for trad)
	# source_times = [[T1, T2],] for Alice's and Bob's electron spins
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

	logger.setLevel(logging.DEBUG)

	# Index generator.
	def get_index():
		j = -1
		while True:
			j += 1
			yield j
	
	index_gen = get_index()
	
	# Misleading names: actually the time interval, not the rate.
	# Note that BASE_CLOCK_RATE is hardcoded to be 10.
	BS_CLOCK_RATE = time_bin # For BeamSplitterProtocol and StateCheckProtocol, for whom time
						 	 # is supposedly of the essence because they require incoming photons
						 	 # to be coincident. ('supposedly' because these protocols
						 	 # do not actually run unprompted, i.e. run_protocol is not run. 
							 # Nevertheless the time step is used as the simultaneity condition.)
	REPEATER_CONTROL_RATE = local_time # How often repeater nodes try BK+2BSM after a failure. 
	NETWORK_RATE = link_time

	# Noise on nuclear spin when electron spin sends a photon.
	if noise_on_nuclear_params is None:
		nna, nnb = 0, 0
	else:
		nna, nnb = noise_on_nuclear_params
	noise_on_nuclear = lambda q: nq.qubitapi.multi_operate(q, [ns.I, ns.X, ns.Y, ns.Z], \
												[1-nna-0.75*nnb, 0.25*nnb, 0.25*nnb, (nna+0.25*nnb)])

	params_list = [BS_CLOCK_RATE, REPEATER_CONTROL_RATE, NETWORK_RATE, \
					source_times, rep_times,
					channel_loss, link_delay,\
					noise_on_nuclear,\
					gate_fidelity, meas_fidelity, prep_fidelity,\
					detector_pdark, detector_eff,\
					repeater_channel_loss, local_delay, reset_delay]

	chain = Chain(index_gen, num_repeaters, num_modes, False)
	chain.set_params(*params_list)
	chain.make_links()
	chain.make_nodes()
	chain.make_planner_control()
	chain.start()

	ns.simutil.sim_run(duration = duration)
	return chain

if __name__ == '__main__':
	chain = run_simulation(3, 2, [[1e9, 1e9],], [[1e9, 1e9], [10e9, 10e9]], 1e-4, duration = 10)

