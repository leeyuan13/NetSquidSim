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
	Protocol to control a traditional repeater.
	Receives commands from a central planner.
	'''
	def __init__(self, time_step, node, reps, conns, planner_conn = None, to_print = False):
		# time_step is irrelevant here.
		# reps = repeater registers
		# conns = list of connections from repeater nodes to the repeater control
		# planner_conn = connection to the central planner
		super().__init__(time_step, node = node, connection = None)
		self.myID = self.node.nodeID

		self.conns = conns
		self.reps = reps

		# status/side = 1 if the repeater node is entangled to the left.
		# status/side = 2 if ... right.
		# First value is for the electron spin; second value is for the nuclear spin.
		self.status = [[0, 0] for _ in range(len(reps))]

		self.planner_conn = planner_conn
		self.to_print = to_print

	def set_planner_conn(self, planner_conn):
		self.planner_conn = planner_conn

	def from_planner_conn(self, args = None):
		# Messages sent from central planner.
		info, _ = self.planner_conn.get_as(self.myID)
		# Possibly multiple instructions.
		for data in info:
			if isinstance(data, tuple) and data[0] == "clock_cycle":
				self.clock_cycle(data)
			elif isinstance(data, tuple) and data[0] == "do_BSM":
				self.do_BSM(data)
			elif isinstance(data, tuple) and data[0] == "unitary":
				self.apply_unitary(data)

	def clock_cycle(self, data):
		# Called at the start of each network clock cycle by the central planner.
		# data[1] = list of (register index, side) to attempt entanglement with
		for index, side in data[1]:
			self.conns[index].put_from(self.myID, data = [('start', side)])
	
	def do_BSM(self, data):
		_, index = data
		self.conns[index].put_from(self.myID, data = [('BSM',)])
	
	def apply_unitary(self, data):
		# Method to run when the central planner wants to apply a unitary based on an adjacent BSM.
		# index = register to apply unitary to
		# isnuclear = whether the qubit is the nuclear spin (otherwise, it is the electron spin)
		_, index, isnuclear, isX, isZ = data
		assert self.status[index][isnuclear] != 0
		if isX == 1: nq.operate(self.reps[index].qmemory.get_qubit(isnuclear), ns.X)
		if isZ == 1: nq.operate(self.reps[index].qmemory.get_qubit(isnuclear), ns.Z)

	def is_entangled(self, index, args = None):
		# Method to run when the repeater registers inform the control that entanglement has been
		# established, either with Alice or with Bob.
		# The index can be determined by setting up the callback-channel linkage appropriately.
		# The side is determined by the message that the repeater node sends.
		[data,], _ = self.conns[index].get_as(self.myID)
		# Entanglement with some qubit.
		if isinstance(data, tuple) and data[0] == "success":
			side = data[1]
			self.status[index][0] = side
			self.planner_conn.put_from(self.myID, data = [('success', index, side)])
			# If nuclear spin is unoccupied, move qubit.
			# Inform the central planner after the qubit has been moved. (*)
			if self.status[index][1] == 0:
				self.conns[index].put_from(self.myID, data = [('move_qubit',)])
			# If the qubit isn't going to be moved, infom the central planner.
			else:
				assert self.status[index][1] != side # check that these are on opposite sides
		# Success of local operation.
		# Note that the SWAP gate is followed by a resetting of the electron spin.
		elif isinstance(data, tuple) and data[0] == "move_qubit_success":
			self.status[index][0], self.status[index][1] = 0, self.status[index][0]
			# Inform the central planner, as in (*).
			self.planner_conn.put_from(self.myID, \
											data = [('move_qubit_success', index)])
		# Success of local Bell state measurements.
		elif isinstance(data, tuple) and data[0] == "BSM_success":
			results = data[2]
			self.status[index][0], self.status[index][1] = 0, 0
			self.planner_conn.put_from(self.myID, data = [('BSM_success', index, results)])
	
class LinkSwitchProtocol(TimedProtocol):
	'''
	Protocol to reroute connections between registers across a link.
	To make better use of the traditional repeater, even only with local information, we must allow
	modes to connect different registers at different times.

	Similar to RepeaterBSProtocol in hybrid12.py.

	The central planner should have direct access to the protocol to add/remove links.
	'''
	def __init__(self, time_step, node, conn1, conn2, test_conn1, test_conn2):
		super().__init__(time_step, node = node, connection = None)
		self.myID = node.nodeID

		# QuantumConnections to/from the repeater registers on both sides.
		# 1 = left registers -- these connections would be side 2 from the registers' perspectives
		# 2 = right registers
		# e.g. self.conn1[i] = connection from index1 = i
		self.conn1 = conn1
		self.conn2 = conn2
		# ClassicalConnection to/from the repeater registers.
		self.test_conn1 = test_conn1
		self.test_conn2 = test_conn2

		# Naive approach: one beamsplitter for each pair of repeater nodes.
		# Connections to/from the beamsplitter.
		# self.repconn1[i][j], self.repconn2[i][j] 
		# 				= connection from the switch node to the constituent beamsplitter used
		#				   when index1 = i and index2 = j are trying to connect
		# self.repconn1[i][j] is the connection for qubits coming from the left bank
		# self.repconn2[i][j] is the connection for qubits coming from the right bank
		self.repconn1 = None
		self.repconn2 = None
		self.test_repconn1 = None
		self.test_repconn2 = None

		# Keep track of which pairs of nodes are active.
		self.from1to2 = dict()
		self.from2to1 = dict()

	def set_repconn(self, repconn1, repconn2, test_repconn1, test_repconn2):
		self.repconn1 = repconn1
		self.repconn2 = repconn2
		self.test_repconn1 = test_repconn1
		self.test_repconn2 = test_repconn2

	def change_switch(self, inst, index1, index2):
		if inst == 'add':
			self.from1to2[index1] = index2
			self.from2to1[index2] = index1
		elif inst == 'remove':
			self.from1to2.pop(index1)
			self.from2to1.pop(index2)
	
	def clear_switch(self):
		self.from1to2 = dict()
		self.from2to1 = dict()
	
	def get_switch(self, side, index):
		if side == 1:
			return self.from1to2.get(index)
		elif side == 2:
			return self.from2to1.get(index)
	
	### Same idea as RepeaterBSProtocol in hybrid12.py.
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
		elif side == 2:
			[msg,], _ = self.test_repconn2[self.from2to1[index]][index].get_as(self.myID)
			self.test_conn2[index].put_from(self.myID, data = [msg,])
		# Note that the repeater registers will inform the repeater control of success/failure directly.

class ScontProtocol(TimedProtocol):
	''' Protocol for source control. '''
	def __init__(self, time_step, node, sources, conns, is_infinite_bank = False, to_print = False):
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID

		self.sources = sources
		self.conns = conns
		
		# Connection to central planner.
		self.planner_conn = None

		# Status = 1 if the source node is entangled with the adjacent 
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
				self.clock_cycle(data)
			elif isinstance(data, tuple) and data[0] == "free":
				# The register is entangled past the first repeater node.
				if self.is_infinite_bank: self.free_up(data)
			elif isinstance(data, tuple) and data[0] == "complete":
				# The register has successfully been entangled across the whole chain.
				if not self.is_infinite_bank: self.free_up(data)
	
	def clock_cycle(self, data):
		# Called at the start of each network clock cycle by the central planner.
		# data[1] = list of register indices to attempt entanglement with
		# Note that source registers only have links on one side.
		for index in data[1]:
			assert self.status[index] == 0
			self.conns[index].put_from(self.myID, data = ['start',])

	def free_up(self, data):
		# Sanity check.
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
			# Inform the central planner.
			self.planner_conn.put_from(self.myID, data = [('success', index),])

class PlannerControlProtocol(TimedProtocol):
	''' 
	Protocol for central planner. 
	Assume a linear (traditional) repeater chain with a defined left and right.
	'''
	def __init__(self, time_step, node, left_scont, right_scont, left_conn, right_conn, rep_conns,\
						m, links, is_infinite_bank = False, to_print = False):
		# time_step = network clock cycle
		# left_scont, right_scont = ScontProtocols for left and right source nodes
		# left_conn, right_conn = ClassicalConnections to source controls
		# rep_conns = ClassicalConnections to repeater controls
		# m = list of m (number of qubits) for each node in rep_conns
		# links = LinkSwitchProtocols for links between nodes
		super().__init__(time_step, node = node, connection = None)
		self.myID = self.node.nodeID

		self.left_scont = left_scont
		self.right_scont = right_scont
		self.left_conn = left_conn
		self.right_conn = right_conn
		self.rep_conns = rep_conns
		self.m = m
		self.links = links
		self.is_infinite_bank = is_infinite_bank
		self.num_rep = len(rep_conns)
		assert len(links) == self.num_rep + 1
		assert len(m) == self.num_rep

		# Keep track of which 'layer' each register is in.
		# When a register completes a BSM, it is moved to the smallest unoccupied layer for the register.
		# Only registers from the same layer will attempt Barrett-Kok.
		# This scheme only utilizes local information, even though it may not be optimal in terms of
		# rate.
		# (An alternative scheme would involve connecting the two longest entanglements, but knowing how
		# long an entanglement is would generally involve global information.)
		# Each register is labeled by (position, index) where position refers to the position of the
		# node in the chain (position = 0 for the left source, position = index + 1 for nodes in
		# rep_conns, position = num_rep+1 for the right source) and the index is labeled by each node.
		# self.register2layer[(position, index)] = (int) layer of the register at (position, index)
		# self.layer2register[(layer, position)] = (int) index of the register in the node
		# self.nextlayer[position] = minimum vacant layer so far
		self.register2layer = dict()
		self.layer2register = dict()
		self.nextlayer = [0]*(self.num_rep+2)
		# Set up these dictionaries.
		for position in range(1, self.num_rep+1):
			for index in range(self.m[position-1]):
				self.register2layer[(position, index)] = index
				self.layer2register[(index, position)] = index
			self.nextlayer[position] = self.m[position-1]
		for index in range(len(self.left_scont.sources)):
			self.register2layer[(0, index)] = index
			self.layer2register[(index, 0)] = index
		self.nextlayer[0] = len(self.left_scont.sources)
		for index in range(len(self.right_scont.sources)):
			self.register2layer[(self.num_rep+1, index)] = index
			self.layer2register[(index, self.num_rep+1)] = index
		self.nextlayer[self.num_rep+1] = len(self.right_scont.sources)

		# self.chain[layer][(position, isnuclear)] = 
		#							[leftmost entangled qubit, rightmost entangled qubit]
		# 							where qubits are encoded as (position, isnuclear)
		# If no left entanglement, then the qubit is labeled as None. (Similarly on the right.)
		# (Indices start from 0, and isnuclear is 0 for the leftmost and rightmost nodes.)
		self.chain = dict()
	
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

		# Whether the chain is waiting for 2BSM+unitaries to be done.
		self.is_holding = False
		self.holding_BSM = deque()

		self.to_print = to_print

	def from_reps(self, position, args = None):
		# position = 1 + index in rep_conns
		info, _ = self.rep_conns[position-1].get_as(self.myID)
		for data in info:
			# Reports of entanglement success.
			if data[0] == 'success':
				_, index, side = data
				layer = self.register2layer[(position, index)]
				if side == 1:
					if position > 1:
						side_qubit = (position-1, 0)
						self.chain.setdefault(layer, dict()).\
								setdefault((position, 0), [None, None])[0] = side_qubit
						# Assume electron spin (i.e. side_qubit[1] = 0) for now.
					elif position == 1:
						# Set both sides at the same time.
						# side_qubit should take the form ('left', 0, index in self.left_qubits).
						side_qubit = ('left', 0, len(self.left_qubits))
						leftindex = self.layer2register[(layer, 0)]
						self.left_qubits.append(self.left_scont.sources[leftindex].qmem.get_qubit(0))
						self.chain.setdefault(layer, dict()).\
								setdefault(side_qubit, [None, None])[1] = (position, 0)
						self.chain[layer].setdefault((position, 0), [None, None])[0] = side_qubit
						# Move to new layer.
						if self.is_infinite_bank:
							self.left_conn.put_from(self.myID, \
											data = [('free', self.layer2register[(layer, 0)])])
							newleftlayer = self.nextlayer[0]
							self.layer2register.pop((layer, 0))
							self.register2layer[(0, leftindex)] = newleftlayer
							self.layer2register[(newleftlayer, 0)] = leftindex
							self.nextlayer[0] = newleftlayer + 1
					# Check if nuclear spin is already entangled on the other side.
					nuclear_sides = self.chain[layer].get((position, 1), [None, None])
					assert nuclear_sides[0] is None
					if nuclear_sides[1] is not None:
						if self.is_holding:
							self.holding_BSM.append((layer, index, position))
						else:
							# Do BSM!
							self.is_holding = True
							self.rep_conns[position-1].put_from(self.myID, data = [('do_BSM', index)])
				elif side == 2:
					if position < self.num_rep:
						side_qubit = (position+1, 0)
						self.chain.setdefault(layer, dict()).\
								setdefault((position, 0), [None, None])[1] = side_qubit
					elif position == self.num_rep:
						side_qubit = ('right', 0, len(self.right_qubits))
						rightindex = self.layer2register[(layer, self.num_rep+1)]
						self.right_qubits.append(self.right_scont.sources[rightindex].qmem.get_qubit(0))
						self.chain.setdefault(layer, dict()).\
								setdefault(side_qubit, [None, None])[0] = (position, 0)
						self.chain[layer].setdefault((position, 0), [None, None])[1] = side_qubit
						# Move to new layer.
						if self.is_infinite_bank:
							self.right_conn.put_from(self.myID, \
									data = [('free', self.layer2register[(layer, self.num_rep+1)])])
							newrightlayer = self.nextlayer[self.num_rep+1]
							rightindex = self.layer2register.pop((layer, self.num_rep+1))
							self.register2layer[(self.num_rep+1, rightindex)] = newrightlayer
							self.layer2register[(newrightlayer, self.num_rep+1)] = rightindex
							self.nextlayer[self.num_rep+1] = newrightlayer + 1
					# Check if nuclear spin is already entangled on the other side.
					nuclear_sides = self.chain[layer].get((position, 1), [None, None])
					assert nuclear_sides[1] is None
					if nuclear_sides[0] is not None:
						if self.is_holding:
							self.holding_BSM.append((layer, index, position))
						else:
							self.is_holding = True
							self.rep_conns[position-1].put_from(self.myID, data = [('do_BSM', index)])
			elif data[0] == 'move_qubit_success':
				_, index = data
				layer = self.register2layer[(position, index)]
				electron_sides = self.chain[layer].pop((position, 0))
				assert (position, 1) not in self.chain[layer] # i.e. nuclear spin should be unoccupied
				self.chain[layer][(position, 1)] = electron_sides
				if electron_sides[0] is not None:
					assert electron_sides[1] is None
					assert self.chain[layer][electron_sides[0]][1] == (position, 0)
					self.chain[layer][electron_sides[0]][1] = (position, 1)
				if electron_sides[1] is not None:
					assert electron_sides[0] is None
					assert self.chain[layer][electron_sides[1]][0] == (position, 0)
					self.chain[layer][electron_sides[1]][0] = (position, 1)
			elif data[0] == 'BSM_success':
				self.is_holding = False
				_, index, results = data
				layer = self.register2layer[(position, index)]
				electron_sides = self.chain[layer].pop((position, 0))
				nuclear_sides = self.chain[layer].pop((position, 1))
				lbell = electron_sides[0]
				if lbell is None: lbell = nuclear_sides[0]
				else: assert nuclear_sides[0] is None
				rbell = electron_sides[1]
				if rbell is None: rbell = nuclear_sides[1]
				else: assert nuclear_sides[1] is None
				# Apply BSM corrections to the left.
				# Note that the qubits at the end are stored in left_qubits and right_qubits.
				if lbell[0] == 'left':
					lqubit = self.left_qubits[lbell[2]] # qubits from the list
														# (works for both infinite and finite banks)
					if results[0] == 1: nq.operate(lqubit, ns.X)
					if results[1] == 1: nq.operate(lqubit, ns.Z)
				else:
					assert lbell[0] > 0
					# Get register index.
					lindex = self.layer2register[(layer, lbell[0])]
					self.rep_conns[lbell[0]-1].put_from(self.myID, \
										data=[("unitary", lindex, lbell[1], results[0], results[1]),])
				# Update connections.
				# If left and right are the ends, then done!
				self.chain[layer][lbell][1] = rbell
				self.chain[layer][rbell][0] = lbell
				if lbell[0] == 'left' and rbell[0] == 'right':
					if self.to_print: print(node, 'complete!', index1, index2)
					# Extract data.
					self.key_times.append(sim_time())
					self.key_dm.append(nq.reduced_dm([self.left_qubits[lbell[2]], \
																self.right_qubits[rbell[2]]]))
					if self.to_print: print(sim_time(), '\n', self.key_dm[-1])
					# Inform nodes.
					self.left_conn.put_from(self.myID, \
											data = [('complete', self.layer2register[(layer, 0)])])
					self.right_conn.put_from(self.myID, \
								data = [('complete', self.layer2register[(layer, self.num_rep+1)])])
					if self.to_print: print(layer, 'complete', sim_time())
					# If infinite bank, these would already have been freed up for other entanglements.
					# If finite bank, free up these registers for further entanglements.
					# Also move into new layers if necessary.
					if not self.is_infinite_bank:
						newleftlayer = self.nextlayer[0]
						leftindex = self.layer2register.pop((layer, 0))
						self.register2layer[(0, leftindex)] = newleftlayer
						self.layer2register[(newleftlayer, 0)] = leftindex
						self.nextlayer[0] = newleftlayer + 1
						newrightlayer = self.nextlayer[self.num_rep+1]
						rightindex = self.layer2register.pop((layer, self.num_rep+1))
						self.register2layer[(self.num_rep+1, rightindex)] = newrightlayer
						self.layer2register[(newrightlayer, self.num_rep+1)] = rightindex
						self.nextlayer[self.num_rep+1] = newrightlayer + 1
					self.chain.pop(layer)
				# Update register2layer and layer2register.
				newlayer = self.nextlayer[position]
				self.register2layer[(position, index)] = newlayer
				self.layer2register.pop((layer, position))
				self.layer2register[(newlayer, position)] = index
				self.nextlayer[position] = newlayer + 1
				# Do next BSM, if necessary.
				if len(self.holding_BSM) > 0:
					_, ii, pp = self.holding_BSM.popleft() # index in node, position
					self.is_holding = True
					self.rep_conns[pp-1].put_from(self.myID, data = [('do_BSM', ii)])
				# Remember to configure the link switches correctly at every clock cycle.

	def run_protocol(self):
		# Set network clock cycle.
		# Decide which side the nodes try.
		# Active registers.
		if self.to_print: print('clock cycle', sim_time())
		active_registers = set(self.layer2register.keys()) # set of (layer, position)
		# Divide registers into layers.
		layered_registers = dict()
		for layer, position in active_registers:
			# Index in the node.
			index = self.layer2register[(layer, position)]
			layered_registers.setdefault(layer, []).append((position, index))
		active_layers = list(layered_registers.keys())
		# Keep track of instructions.
		# instructions[position] = list of (index, side) to attempt entanglement with
		instructions = [[] for _ in range(self.num_rep+2)]
		# Clear switches.
		for j in range(self.num_rep+1):
			self.links[j].clear_switch()
		# For each layer, identify the available links.
		for layer in active_layers:
			layered_registers[layer].sort(key = lambda x: x[0]) # sort by position
			# Identify which links are available in the layer.
			# Links are labeled by the position of their left node, so the first link has label 0.
			# We try Barrett-Kok on the link if (1) it is available, and
			#								    (2) the link to the left has not been tried.
			# A link is available if (1) the two registers it connects are in the same layer,
			#						 (2) the left-side register has no entanglement to the right, and
			#						 (3) the right-side register has no entanglement to the left.
			last_tried_link = None # label for last tried link
			for i in range(len(layered_registers[layer])-1):
				# Check if the link to the left has been tried.
				left_register = layered_registers[layer][i]
				if last_tried_link is not None and left_register[0] == last_tried_link + 1: continue
				# Check if available.
				next_register = layered_registers[layer][i+1]
				# If the next register is not the right-side register of the link, then move on.
				if next_register[0] != left_register[0] + 1: continue
				# Otherwise, check if there is already entanglement.
				# Note that the leftmost registers are coded as
				# ('left', isnuclear, index in self.left_qubits); similarly for the rightmost registers.
				# Hence, we need to check if an entanglement link already exists from the inner
				# repeater registers.
				if left_register[0] != 0:
					if self.chain.get(layer,dict()).get((left_register[0],0),[None,None])[1] \
							is not None \
				  			or self.chain.get(layer,dict()).get((left_register[0],1),[None,None])[1] \
							is not None:
						continue
				else: # for the left link, need to check from the right register
					if self.chain.get(layer,dict()).get((next_register[0],0),[None,None])[0] \
							is not None \
							or self.chain.get(layer,dict()).get((next_register[0],1),[None,None])[0] \
							is not None:
						continue
				assert self.chain.get(layer, dict()).get((left_register[0], 0), [None, None])[1] \
					is None, (layer, left_register, self.chain[layer][(left_register[0], 0)])
				assert self.chain.get(layer, dict()).get((left_register[0], 1), [None, None])[1] \
					is None, (layer, left_register, self.chain[layer][(left_register[0], 1)])
				assert self.chain.get(layer, dict()).get((next_register[0], 0), [None, None])[0] \
					is None, (layer, left_register, self.chain[layer][(next_register[0], 0)])
				assert self.chain.get(layer, dict()).get((next_register[0], 1), [None, None])[0] \
					is None, (layer, left_register, self.chain[layer][(next_register[0], 1)])
				# If the link is free, then set link switches.
				self.links[left_register[0]].change_switch('add', left_register[1], next_register[1])
				# Now update instructions.
				instructions[left_register[0]].append((left_register[1], 2))
				instructions[next_register[0]].append((next_register[1], 1))
				last_tried_link = left_register[0]
		# Inform source and repeater nodes.
		self.left_conn.put_from(self.myID, \
							data = [('clock_cycle', [ind[0] for ind in instructions[0]]),])
		self.right_conn.put_from(self.myID, \
							data = [('clock_cycle', [ind[0] for ind in instructions[self.num_rep+1]]),])
		for j in range(1, self.num_rep+1):
			self.rep_conns[j-1].put_from(self.myID, \
							data = [('clock_cycle', instructions[j]),])
		
class TraditionalRepeater:
	''' 
	Registers associated with a traditional repeater node. 
	The code implicitly assumes that the traditional repeater lies in a chain.
	'''
	def __init__(self, index_gen, num_qubits, left_correct = True, right_correct = False):
		# index_gen = (global) index generator
		# num_qubits = degree of parallelism, m (i.e. number of total registers)
		self.index_gen = index_gen
		self.num_qubits = num_qubits
		# Whether to correct for BK phase.
		self.left_correct = left_correct # correct for link on the left?
		self.right_correct = right_correct # correct for link on the right?

		### NetSquid nodes
		# Registers.
		self.regs = None
		# Repeater control.
		self.rep_control = None

		### NetSquid connections
		# Connections to the left.
		self.left_conns = None
		self.left_test_conns = None
		# To the right.
		self.right_conns = None
		self.right_test_conns = None
		# LongDistanceLink objects.
		self.left_link = None
		self.right_link = None
		
		### NetSquid protocols
		# Protocols for registers.
		self.protos = None
		# Protocol for repeater control.
		self.rep_control_prot = None

		### Parameters
		# As usual, note that rates are actually time intervals.
		self.BASE_CLOCK_RATE = 1e6
		# Coherence times: a list of [T1, T2], the length of which is equal to the number of memories
		self.rep_times = None
		# Interaction noise on nuclear spin.
		self.noise_on_nuclear = None
		# Fidelities.
		self.gate_fidelity = None
		self.meas_fidelity = None
		self.prep_fidelity = None
		# Reset delay between pulses in a Barrett-Kok sequence.
		self.reset_delay = None
		# Whether parameters have been set.
		self.params = False
	
	def set_params(self, rep_times, noise_on_nuclear, gate_fidelity, meas_fidelity, prep_fidelity,\
						reset_delay):
		self.rep_times = rep_times
		self.noise_on_nuclear = noise_on_nuclear
		self.gate_fidelity = gate_fidelity
		self.meas_fidelity = meas_fidelity
		self.prep_fidelity = prep_fidelity
		self.reset_delay = reset_delay

		self.params = True
		# Next steps:
		#	make registers (NetSquid nodes)
		#	(set up Barrett-Kok links on both sides -- not here)
		# 	make protocols for registers (which have the aforementioned links as prerequisites)
		#	(inform link switch, set up node protocols for link switch)
		# 	make repeater control and its protocol

	def set_links(self, left_link, right_link):
		self.left_link = left_link
		self.right_link = right_link
		self.left_conns = left_link.conn2
		self.right_conns = right_link.conn1
		self.left_test_conns = left_link.test_conn2
		self.right_test_conns = right_link.test_conn1
		assert len(self.left_conns) == self.num_qubits
		assert len(self.right_conns) == self.num_qubits
		assert len(self.left_conns) == len(self.left_test_conns)
		assert len(self.right_conns) == len(self.right_test_conns)

	def make_registers(self):
		assert self.params
		assert self.regs is None
		self.regs = []
		for i in range(self.num_qubits):
			index = self.index_gen.__next__()
			atoms = StandardMemoryDevice("atom"+str(index), 2, decoherenceTimes = self.rep_times)
			reg_node = QuantumNode("source"+str(index), index, memDevice = atoms)
			self.regs.append(reg_node)

	def make_register_protocols(self):
		# Make protocols for registers.
		assert self.left_conns is not None
		assert self.right_conns is not None
		assert self.protos is None

		self.protos = []
		for i in range(self.num_qubits):
			next_prot = BellProtocol(self.BASE_CLOCK_RATE, self.regs[i], \
								self.left_conns[i], self.left_test_conns[i], \
								other_conn = self.right_conns[i], \
								test_other_conn = self.right_test_conns[i], \
								to_correct = (self.left_correct, self.right_correct), \
								noise_on_nuclear = self.noise_on_nuclear, \
								gate_fidelity = self.gate_fidelity, \
								meas_fidelity = self.meas_fidelity, \
								prep_fidelity = self.prep_fidelity, \
								reset_delay = self.reset_delay)
			self.regs[i].add_node_protocol(next_prot)
			self.regs[i].start()
			self.protos.append(next_prot)

	def make_rep_control(self):
		assert self.protos is not None
		assert self.rep_control is None
		assert self.rep_control_prot is None

		index = self.index_gen.__next__()
		self.rep_control = QuantumNode("rep_control"+str(index), index)

		# Make connections with repeater control.
		# Repeater protocols should have a set_repeater_conn method that takes a ClassicalConnection
		# to rep_control as input.
		repconns = []
		for i in range(self.num_qubits):
			next_repconn = ClassicalConnection(self.regs[i], self.rep_control)
			self.protos[i].set_repeater_conn(next_repconn)
			repconns.append(next_repconn)

		# Set up repeater control protocol.
		# The repeater control protocol should have an is_entangled method that is run every time a node
		# is successfully entangled by Barrett-Kok.
		self.rep_control_prot = RepeaterControlProtocol(self.BASE_CLOCK_RATE, self.rep_control, \
										self.regs, repconns)
		# Set up handlers.
		for i in range(self.num_qubits):
			func = lambda args, index = i: self.rep_control_prot.is_entangled(index, args)
			repconns[i].register_handler(self.regs[i].nodeID, self.protos[i].start_BK)
			repconns[i].register_handler(self.rep_control.nodeID, func)
		self.rep_control.add_node_protocol(self.rep_control_prot)

class ClientNode:
	''' Registers, connections and protocols associated with a client node. '''
	def __init__(self, index_gen, num_qubits, is_infinite_bank = False):
		# index_gen = (global) index generator
		# num_qubits = degree of parallelism, m
		# is_infinite_bank = whether the client node has (effectively) an infinite number of registers
		#					 (or whether they only have a finite bank of num_qubits registers)
		self.index_gen = index_gen
		self.num_qubits = num_qubits
		self.is_infinite_bank = is_infinite_bank

		### NetSquid nodes
		# Registers.
		self.regs = None
		# Source control.
		self.scont = None

		### NetSquid protocols
		# Protocols for registers.
		self.protos = None
		# Protocol for scont.
		self.scont_prot = None

		### NetSquid connections
		self.conns = None
		self.test_conns = None
		# LongDistanceLink object.
		self.link = None
		# Whether the client node is on the right of the chain.
		self.is_right = None

		### Parameters
		# As usual, note that rates are actually time intervals.
		self.BASE_CLOCK_RATE = 1e6
		self.atom_times = None
		self.prep_fidelity = None
		self.reset_delay = None

	def set_params(self, atom_times, prep_fidelity, reset_delay):
		# atom_times = list of [T1, T2] for each qubit in a register
		# In this case, len(atom_times) = 1.
		self.atom_times = atom_times
		self.prep_fidelity = prep_fidelity
		self.reset_delay = reset_delay

	def set_link(self, link, is_right):
		self.link = link # LongDistanceLink object
		self.is_right = is_right # whether the node is on the right of the link,
								 # i.e. whether the link is on the left of the node
		self.conns = link.conn1 if not is_right else link.conn2
		self.test_conns = link.test_conn1 if not is_right else link.test_conn2
		assert len(self.conns) == self.num_qubits
		assert len(self.conns) == len(self.test_conns)

	def make_registers(self):
		assert self.atom_times is not None
		assert self.regs is None
		self.regs = []
		for i in range(self.num_qubits):
			index = self.index_gen.__next__()
			atoms = StandardMemoryDevice("atom"+str(index), 1, decoherenceTimes = self.atom_times)
			reg_node = QuantumNode("source"+str(index), index, memDevice = atoms)
			self.regs.append(reg_node)
		# Idea: make registers
		#		(make link, not here)
		#		make register protocols
		#		make scont and its protocol

	def make_register_protocols(self):
		assert self.conns is not None
		assert self.test_conns is not None
		assert self.protos is None

		self.protos = []
		for i in range(self.num_qubits):
			# By convention, the register on the right side of the link does the BK phase correction.
			next_prot = SourceProtocol(self.BASE_CLOCK_RATE, self.regs[i], self.conns[i], \
											self.test_conns[i], to_correct = self.is_right, \
											prep_fidelity = self.prep_fidelity,\
											reset_delay = self.reset_delay)
			self.regs[i].add_node_protocol(next_prot)
			self.regs[i].start()
			self.protos.append(next_prot)

	def make_scont(self):
		assert self.protos is not None
		assert self.scont is None
		assert self.scont_prot is None

		index = self.index_gen.__next__()
		self.scont = QuantumNode("scont"+str(index), index)

		# Make connections with scont.
		scont_conns = []
		for i in range(self.num_qubits):
			next_conn = ClassicalConnection(self.scont, self.regs[i])
			self.protos[i].set_scont_conn(next_conn)
			scont_conns.append(next_conn)
			next_conn.register_handler(self.regs[i].nodeID, self.protos[i].start_BK)

		# Set up scont protocol.
		self.scont_prot = ScontProtocol(self.BASE_CLOCK_RATE, self.scont, self.regs, scont_conns, \
											self.is_infinite_bank)
		for i in range(self.num_qubits):
			func = lambda args, index = i: self.scont_prot.is_entangled(index, args)
			scont_conns[i].register_handler(self.scont.nodeID, func)
		self.scont.add_node_protocol(self.scont_prot)

class LongDistanceLink:
	''' 
	Registers, connections, protocols associated with a long-distance link between adjacent
	repeater nodes.
	Programatically, repeater nodes are the primary object in a repeater chain, and adjacent
	nodes are connected by links. Each link has multiple modes, and each mode has an associated 
	entanglement protocol (in this case, Barrett-Kok).
	The link contains a link switch to match different registers on either side of the link.
	The network has an implicit left-to-right directionality.

	Convention: the register on the right side of the Barrett-Kok link will do the 
				BK phase correction.
	'''
	def __init__(self, index_gen):
		# index_gen = (global) index generator
		self.index_gen = index_gen

		### NetSquid nodes
		self.left_regs = None
		self.right_regs = None

		### NetSquid protocols
		self.left_protos = None
		self.right_protos = None

		### Connections between nodes
		# 1 = left, 2 = right
		# These connections go from the adjacent repeater registers to the link switch.
		self.conn1 = None
		self.conn2 = None
		self.test_conn1 = None
		self.test_conn2 = None

		### Link switch
		self.link_switch_node = None
		self.link_switch_prot = None
		# No need to keep the beamsplitters in the link switch.

		### Relevant link parameters.
		# Clock rates.
		self.BASE_CLOCK_RATE = 1e6 # for nodes that don't rely on frequent checks, 
								   # especially reactive protocols
		self.BS_CLOCK_RATE = None  # for BeamSplitterProtocol and StateCheckProtocol; defines
								   # whether two photons arrived simultaneously
		
		self.channel_loss = None
		self.delay = None
		self.detector_pdark = None
		self.detector_eff = None
		# Whether the params have been set.
		self.params = False
	
	def set_registers(self, left_regs, right_regs):
		self.left_regs = left_regs # NetSquid nodes, not custom objects
		self.right_regs = right_regs

	def set_protocols(self, left_protos, right_protos):
		self.left_protos = left_protos
		self.right_protos = right_protos
	
	def set_params(self, BS_CLOCK_RATE, channel_loss, delay, detector_pdark, detector_eff):
		self.BS_CLOCK_RATE = BS_CLOCK_RATE
		self.channel_loss = channel_loss
		self.delay = delay
		self.detector_pdark = detector_pdark
		self.detector_eff = detector_eff
		self.params = True
		
		# Repeater registers to link switch.
		self.node_to_ls = lambda x, y: QuantumConnection(x, y, \
											noise_model = QubitLossNoiseModel(channel_loss),\
											delay_model = FixedDelayModel(delay=delay))
		# Beam splitter to link switch.
		self.bs_to_ls = lambda x, y: QuantumConnection(x, y)
		# Beam splitter to detectors. 
		self.bs_to_det = lambda x, y: QuantumConnection(x, y)
		
		# x, y, z, a, b = node, conn1, conn2, test_conn1, test_conn2
		self.make_ls_prot = lambda x, y, z, a, b: \
											LinkSwitchProtocol(self.BASE_CLOCK_RATE, x, y, z, a, b)
		self.make_bs_prot = [lambda x, y, z: BeamSplitterProtocol(BS_CLOCK_RATE, x, y, z),\
							 lambda x, y, z: StateCheckProtocol(BS_CLOCK_RATE, x, y, z)]
		self.make_det_prot = lambda x, y, z: DetectorProtocol(self.BASE_CLOCK_RATE, x, y, z, \
												to_print = False, \
												pdark = detector_pdark, efficiency = detector_eff)
	
	def make_link_switch(self):
		assert self.params
		assert self.link_switch_node is None
		assert self.link_switch_prot is None

		next_index = self.index_gen.__next__()
		self.link_switch_node = QuantumNode("ls"+str(next_index), next_index)

		# First establish connections from the registers to the link switch.
		conn1 = []
		conn2 = []
		test_conn1 = []
		test_conn2 = []
		for i in range(len(self.left_regs)):
			next_conn = self.node_to_ls(self.left_regs[i], self.link_switch_node)
			next_test_conn = ClassicalConnection(self.link_switch_node, self.left_regs[i],\
											delay_model = FixedDelayModel(delay=self.delay))
			conn1.append(next_conn)
			test_conn1.append(next_test_conn)
		for i in range(len(self.right_regs)):
			next_conn = self.node_to_ls(self.right_regs[i], self.link_switch_node)
			next_test_conn = ClassicalConnection(self.link_switch_node, self.right_regs[i],\
											delay_model = FixedDelayModel(delay=self.delay))
			conn2.append(next_conn)
			test_conn2.append(next_test_conn)

		self.conn1 = conn1
		self.conn2 = conn2
		self.test_conn1 = test_conn1
		self.test_conn2 = test_conn2

		# Make link switch protocol.
		# The link switch should be controlled directly by the central planner.
		self.link_switch_prot = self.make_ls_prot(self.link_switch_node, \
																conn1, conn2, test_conn1, test_conn2)
		
		# Establish connections from the link switch to the constituent beamsplitters.
		repconn1 = []
		repconn2 = []
		test_repconn1 = []
		test_repconn2 = []
		for i in range(len(self.left_regs)):
			repconn1.append([])
			repconn2.append([])
			test_repconn1.append([])
			test_repconn2.append([])
			for j in range(len(self.right_regs)):
				# Create new beamsplitters, detectors etc.		
				index = self.index_gen.__next__()
				detector1 = QuantumNode("detector"+str(index), index)
				index = self.index_gen.__next__()
				detector2 = QuantumNode("detector"+str(index), index)
				# Beamsplitter.
				index = self.index_gen.__next__()
				beamsplitter = QuantumNode("beamsplitter"+str(index), index)
				# Quantum connections from link switch to constituent beamsplitters.
				next_conn1 = self.bs_to_ls(self.link_switch_node, beamsplitter)
				next_conn2 = self.bs_to_ls(self.link_switch_node, beamsplitter)
				# Quantum connections from constituent beamsplitters to detectors.
				next_conn3 = self.bs_to_det(beamsplitter, detector1)
				next_conn4 = self.bs_to_det(beamsplitter, detector2)
				# Classical connections.
				next_test_conn1 = ClassicalConnection(detector1, beamsplitter)
				next_test_conn2 = ClassicalConnection(detector2, beamsplitter)
				next_test_conn3 = ClassicalConnection(beamsplitter, self.link_switch_node)
				next_test_conn4 = ClassicalConnection(beamsplitter, self.link_switch_node)
				# Set up protocols at detectors and the constituent beamsplitters.
				next_proto3 = self.make_det_prot(detector1, next_conn3, next_test_conn1)
				next_proto4 = self.make_det_prot(detector2, next_conn4, next_test_conn2)
				BSproto = self.make_bs_prot[0](beamsplitter, [next_conn1, next_conn2], \
											[next_conn3, next_conn4])
				SCproto = self.make_bs_prot[1](beamsplitter, [next_test_conn1, next_test_conn2], \
											[next_test_conn3, next_test_conn4])
				# Register handlers for beamsplitters and detectors.
				next_conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
				next_conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
				next_test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
				next_test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)
				detector1.setup_connection(next_conn3, [next_proto3])
				detector2.setup_connection(next_conn4, [next_proto4])
				# Register handlers for link switch (for messages to/from constituent beamsplitters).
				func = lambda args, side = 1, index = i: \
										self.link_switch_prot.from_bs(side, index, args)
				next_test_conn3.register_handler(self.link_switch_node.nodeID, func)
				func = lambda args, side = 2, index = j: \
										self.link_switch_prot.from_bs(side, index, args)
				next_test_conn4.register_handler(self.link_switch_node.nodeID, func)

				# Add connections to the list.
				repconn1[i].append(next_conn1)
				repconn2[i].append(next_conn2)
				test_repconn1[i].append(next_test_conn3)
				test_repconn2[i].append(next_test_conn4)

		# Set these connections up so that the link switch can respond.
		self.link_switch_prot.set_repconn(repconn1, repconn2, test_repconn1, test_repconn2)

	def set_node_protocols(self):
		# Establish handlers with adjacent registers after protocols have been set up.
		for i in range(len(self.left_regs)):
			func = lambda arg, side = 1, index = i: \
										self.link_switch_prot.from_rep_registers(side, index, arg)
			self.conn1[i].register_handler(self.link_switch_node.nodeID, func)
			func2 = lambda arg, side = 2, index = i: \
										self.left_protos[index].verification(side, arg)
			self.test_conn1[i].register_handler(self.left_regs[i].nodeID, func2)
		for i in range(len(self.right_regs)):
			func = lambda arg, side = 2, index = i: \
										self.link_switch_prot.from_rep_registers(side, index, arg)
			self.conn2[i].register_handler(self.link_switch_node.nodeID, func)
			func2 = lambda arg, side = 1, index = i: \
										self.right_protos[index].verification(side, arg)
			self.test_conn2[i].register_handler(self.right_regs[i].nodeID, func2)

class Chain:
	'''
	Contains the full repeater chain, including the central planner.
	Assume that all long-distance links have the same number of modes.
	'''
	def __init__(self, index_gen, num_repeaters, num_modes, is_infinite_bank):
		self.index_gen = index_gen # (global) index generator
		self.num_repeaters = num_repeaters # number of repeater nodes, excluding the client nodes
		self.num_modes = num_modes
		self.is_infinite_bank = is_infinite_bank
	
		### Nodes objects.
		self.left_node = None # ClientNode object
		self.right_node = None
		self.rep_nodes = None # list of TraditionalRepeater objects

		### Long-distance links.
		self.links = None # list of LongDistanceLink objects

		### Central planner.
		self.planner_control = None # NetSquid node
		self.planner_control_prot = None # PlannerControlProtocol object

		### Relevant parameters.
		# Clock rates.
		# Recall that rates are actually time intervals.
		# BASE_CLOCK_RATE is hardcoded to be 1e6.
		self.BS_CLOCK_RATE = None
		self.NETWORK_RATE = None
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
		# Atom property.
		self.reset_delay = None
		# Whether params have been set up.
		self.is_params = False

	# Setting up the network:
	# 	TraditionalRepeater and ClientNode objects generate the required registers (= NetSquid nodes).
	# 	LongDistanceLink objects contain the individual connections that make up a link.
	# 	We then set up the protocols for the registers, and the protocols for the link switches
	#	in each link.
	#	Then we set up the central planner.
	#	(Neglect communication times with the planner for simplicity.)

	def set_params(self, BS_CLOCK_RATE, NETWORK_RATE, \
						source_times, rep_times,\
						channel_loss, link_delay,\
						noise_on_nuclear,\
						gate_fidelity, meas_fidelity, prep_fidelity,\
						detector_pdark, detector_eff,\
						reset_delay):
		self.BS_CLOCK_RATE		= BS_CLOCK_RATE
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
		self.reset_delay		= reset_delay
		
		self.is_params = True

	def make_nodes(self):
		assert self.left_node is None
		assert self.right_node is None
		assert self.rep_nodes is None
		assert self.is_params

		self.left_node = ClientNode(self.index_gen, self.num_modes, self.is_infinite_bank)
		self.left_node.set_params(self.source_times, self.prep_fidelity, self.reset_delay)
		self.left_node.make_registers()
		self.right_node = ClientNode(self.index_gen, self.num_modes, self.is_infinite_bank)
		self.right_node.set_params(self.source_times, self.prep_fidelity, self.reset_delay)
		self.right_node.make_registers()
		
		self.rep_nodes = []
		lc, rc = False, True # whether to do BK phase correction on the left/right of a link
		for i in range(self.num_repeaters):
			# Note that the left of a repeater node is the right of a link.
			next_node = TraditionalRepeater(self.index_gen, self.num_modes, rc, lc)
			next_node.set_params(self.rep_times, self.noise_on_nuclear, \
									self.gate_fidelity, self.meas_fidelity, self.prep_fidelity,\
									self.reset_delay)
			next_node.make_registers()
			self.rep_nodes.append(next_node)
	
	def make_links(self):
		assert self.rep_nodes is not None
		assert self.left_node is not None
		assert self.right_node is not None
		assert self.links is None

		self.links = []
		left_link = LongDistanceLink(self.index_gen)
		left_link.set_params(self.BS_CLOCK_RATE, self.channel_loss, self.link_delay, \
									self.detector_pdark, self.detector_eff)
		left_link.set_registers(self.left_node.regs, self.rep_nodes[0].regs)
		left_link.make_link_switch()
		self.links.append(left_link)
		for i in range(self.num_repeaters-1):
			next_link = LongDistanceLink(self.index_gen)
			next_link.set_params(self.BS_CLOCK_RATE, self.channel_loss, self.link_delay, \
									self.detector_pdark, self.detector_eff)
			next_link.set_registers(self.rep_nodes[i].regs, self.rep_nodes[i+1].regs)
			next_link.make_link_switch()
			self.links.append(next_link)
		right_link = LongDistanceLink(self.index_gen)
		right_link.set_params(self.BS_CLOCK_RATE, self.channel_loss, self.link_delay, \
								self.detector_pdark, self.detector_eff)
		right_link.set_registers(self.rep_nodes[self.num_repeaters-1].regs, self.right_node.regs)
		right_link.make_link_switch()
		self.links.append(right_link)

		# Complete the process of making register protocols and scont/repeater controls.
		self.left_node.set_link(self.links[0], False)
		self.left_node.make_register_protocols()
		self.left_node.make_scont()
		for i in range(self.num_repeaters):
			self.rep_nodes[i].set_links(self.links[i], self.links[i+1])
			self.rep_nodes[i].make_register_protocols()
			self.rep_nodes[i].make_rep_control()
		self.right_node.set_link(self.links[self.num_repeaters], True)
		self.right_node.make_register_protocols()
		self.right_node.make_scont()

		# Match connections to node / link switch protocols.
		self.links[0].set_protocols(self.left_node.protos, self.rep_nodes[0].protos)
		self.links[0].set_node_protocols()
		for i in range(self.num_repeaters-1):
			self.links[i+1].set_protocols(self.rep_nodes[i].protos, self.rep_nodes[i+1].protos)
			self.links[i+1].set_node_protocols()
		self.links[self.num_repeaters].\
					set_protocols(self.rep_nodes[self.num_repeaters-1].protos, self.right_node.protos)
		self.links[self.num_repeaters].set_node_protocols()
		
		text = str(self.left_node.scont.nodeID) + '--'
		for i in range(self.num_repeaters):
			text += str(self.rep_nodes[i].rep_control.nodeID) + '--'
		text += str(self.right_node.scont.nodeID)
		print(text)

	def make_planner_control(self):
		assert self.links is not None
		assert self.planner_control is None
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
		self.planner_control_prot = PlannerControlProtocol(self.NETWORK_RATE, self.planner_control, \
										self.left_node.scont_prot, self.right_node.scont_prot, \
										left_conn, right_conn, rep_conns, \
										[self.num_modes]*self.num_repeaters, \
										# the planner control requires LinkSwitchProtocol objects,
										# which are contained in LongDistanceLink objects
										[ll.link_switch_prot for ll in self.links], \
										self.is_infinite_bank)
		# Set up handlers.
		left_conn.register_handler(self.left_node.scont.nodeID, \
													self.left_node.scont_prot.from_planner_conn)
		right_conn.register_handler(self.right_node.scont.nodeID, \
													self.right_node.scont_prot.from_planner_conn)
		# The leftmost and rightmost nodes don't talk to the planner control.
		for i in range(self.num_repeaters):
			rep_conns[i].register_handler(self.rep_nodes[i].rep_control.nodeID,\
												self.rep_nodes[i].rep_control_prot.from_planner_conn)
			func = lambda args, position = i+1: self.planner_control_prot.from_reps(position, args)
			rep_conns[i].register_handler(self.planner_control.nodeID, func)
		self.planner_control.add_node_protocol(self.planner_control_prot) 
		# Necessary for self.planner_control.start() to activate run_protocol().

	def start(self):
		self.planner_control.start()

def run_simulation(num_repeaters, num_modes, source_times, rep_times, channel_loss, duration = 100, \
					repeater_channel_loss = None, noise_on_nuclear_params = None, \
					link_delay = 0., link_time = 1., local_delay = None, local_time = None, \
					time_bin = 0.01, detector_pdark = 1e-7, detector_eff = 0.93,\
					gate_fidelity = 0.999, meas_fidelity = 0.9998, prep_fidelity = 0.99, \
					reset_delay = 0.1):
	## Refer to hybrid12.py.
	## Parameters:
	# num_repeaters = number of repeaters in the hybrid chain
	# num_modes = degree of parallelism (m/2 for hybrid, m for trad)
	# source_times = [[T1, T2],] for Alice's and Bob's electron spins
	# rep_times = [[T1, T2] for electron spin, [T1, T2] for nuclear spin]
	# channel_loss = probability of losing a photon between a client node and a detector station
	# duration = length of simulation in nanoseconds
	# noise_on_nuclear_params = [a = 1/4000, b = 1/5000] parameters for the depolarizing and dephasing
	#								noise experienced by the nuclear spin when the electron spin sends
	#								a photon
	## Temporal parameters (in nanoseconds):
	# link_delay = time of flight from a node to a detector station
	# 				(Barrett-Kok requires (at least) 4 time-of-flights)
	# link_time = network clock cycle (distant Barrett-Kok attempts are prompted once every 
	# 									 link_time, so link_time >= 4 link_delay)
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
	
	## Note that repeater_channel_loss is irrelevant here, because there are no channels
	## within the repeater.
	## Similarly, local_delay and local_time mean nothing here.
	
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
	# Note that BASE_CLOCK_RATE is hardcoded to be 1e6.
	BS_CLOCK_RATE = time_bin # For BeamSplitterProtocol and StateCheckProtocol, for whom time
						 	 # is supposedly of the essence because they require incoming photons
						 	 # to be coincient. ('supposedly' because these protocols
						 	 # do not actually run unprompted, i.e. run_protocol is not run.)
	NETWORK_RATE = link_time # How often the central planner sends start signals to eligible
							 # repeater nodes.
	
	# Noise on nuclear spin when electron spin sends a photon.
	if noise_on_nuclear_params is None:
		nna, nnb = 0, 0
	else:
		nna, nnb = noise_on_nuclear_params
	noise_on_nuclear = lambda q: nq.qubitapi.multi_operate(q, [ns.I, ns.X, ns.Y, ns.Z], \
													[1-nna-0.75*nnb, 0.25*nnb, 0.25*nnb, (nna+0.25*nnb)])

	params_list = [BS_CLOCK_RATE, NETWORK_RATE, \
					source_times, rep_times,
					channel_loss, link_delay,\
					noise_on_nuclear,\
					gate_fidelity, meas_fidelity, prep_fidelity,\
					detector_pdark, detector_eff,\
					reset_delay]

	chain = Chain(index_gen, num_repeaters, num_modes, False)
	chain.set_params(*params_list)
	chain.make_nodes() # make nodes for a traditional chain first
	chain.make_links()
	chain.make_planner_control()
	chain.start()

	ns.simutil.sim_run(duration = duration)
	return chain

if __name__ == '__main__':
	import time
	start_time = time.time()
	#chain = run_simulation(3, 2, [[1e9, 1e9],], [[1e9, 1e9], [10e9, 10e9]], 1e-4, duration = 10)
	#chain = run_simulation(1, 10, [[1e9, 1e9],], [[1e9, 1e9], [10e9, 10e9]], 1e-4, duration = 20)
	chain = run_simulation(3, 2, [[1e9, 1e9],], [[1e9, 1e9], [10e9, 10e9]], 1e-4, duration = 1e2,\
							link_delay = 1, link_time = 5, local_delay = 1e-2, local_time = 1e-1, \
							time_bin = 1e-5, reset_delay = 1e-4)
	end_time = time.time()
	print(end_time - start_time)


