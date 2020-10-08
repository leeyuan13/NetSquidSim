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

from bk9 import AtomProtocol, BeamSplitterProtocol, DetectorProtocol,\
				StateCheckProtocol, QubitLossNoiseModel
# Use BSM3 because it allows for imperfect measurements.
from BSM3 import BellStateMeasurement as BSM
from BSM3 import depolarization

np.set_printoptions(precision=2)

class SourceProtocol(AtomProtocol):
	''' Protocol for source registers. '''
	def __init__(self, time_step, node, connection, test_conn, to_correct = False, \
					prep_fidelity = 1., reset_delay = 0.01):
		super().__init__(time_step, node, connection, test_conn, to_run = False, \
						 to_correct = to_correct, prep_fidelity = prep_fidelity, \
						 reset_delay = reset_delay)
		# time_step should not matter because SourceProtocol is reactive (i.e. only acts when
		# it receives a message).
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
		
	def verification(self, side = None, args = None):
		# Do the Barrett-Kok procedure.
		# Add the side argument for convenience, to be consistent with BellProtocol.
		# (See e.g. hybrid12.LongDistanceLink.create_BK.)
		super().verification(args)
		# Also inform the source control that entanglement is successful.
		# Note that the source registers only send single strings as classical messages.
		if self.stage == 3 and self.scont_conn is not None:
			self.scont_conn.put_from(self.myID, data = ['success',])

	def run_protocol(self):
		pass

class BellProtocol(AtomProtocol):
	''' Protocol for repeater registers. '''
	def __init__(self, time_step, node, connection, test_conn, other_conn = None, \
							test_other_conn = None, to_correct = False, to_print = False,\
							noise_on_nuclear = None, gate_fidelity = 1., meas_fidelity = 1.,\
							prep_fidelity = 1., reset_delay = 0.01):
		# time_step should not matter because BellProtocol, like SourceProtocol, is reactive.
		# connection, test_conn are the quantum and classical connections (repectively) to adjacent
		# registers. Similarly for other_conn, test_other_conn.
		# to_correct tells us whether the protocol should correct for phase in Barrett-Kok.
		# noise_on_nuclear is the noise to be applied on the nuclear spin every time the electron spin is
		#	used; it comprises a dephasing and depolarization, see Rozpedek et al.
		# gate_fidelity is the fidelity of a single 2-qubit gate.
		# meas_fidelity is the fidelity with which the electron spin can be be measured -- affects
		# 	the fidelity of Bell state measurements.
		# Note that "node" in the NetSquid sense is different from the use of "node" in the paper.
		# 	A NetSquid node controls something -- here, it controls a register, but it can also
		# 	send messages to control an array of registers, etc.
		# 	A physical node is a colocated collection of registers, with associated switches,
		#	connections, etc.
		super().__init__(time_step, node, connection, test_conn, to_run = False, \
						 to_correct = to_correct if isinstance(to_correct, bool) else False, \
						 prep_fidelity = prep_fidelity, \
						 reset_delay = reset_delay)
		# Note that the quantum memory has 2 atoms now:
		#	electron spin (active) = index 0
		# 	nuclear spin (storage) = index 1
		self.repeater_conn = None # classical connection to repeater control
		self.other_conn = other_conn # quantum connection to repeater beamsplitter
		self.test_other_conn = test_other_conn # classical connection to repeater beamsplitter
		self.num_electron_attempts = 0 # number of photons sent by the electron spin before a
									   # successful entanglement is achieved.
									   # This is important because it may affect the nuclear spin.

		# To keep track of which detector observed the photon.
		self.detector_order = None
		# Note that self.verification is rewritten, so we can let to_correct be a tuple of
		# (bool, bool) indicating sides 1 and 2.
		self.to_correct = to_correct if isinstance(to_correct, tuple) else (to_correct, to_correct)

		# To keep track of which side Barrett-Kok needs to be done on.
		# 0 = no Barrett-Kok
		# 1 = Barrett-Kok on side 1
		# 2 = Barrett-Kok on other side
		self.side = 0

		# Whether to print output.
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

	# Override self.send_photon(), since either connection could be used.
	def send_photon(self):
		# Send photon down self.side, since it is used in run_protocol_inner and run_protocol_later.
		if self.side == 1:
			self.conn.put_from(self.myID, data = [self.photon])
		elif self.side == 2:
			self.other_conn.put_from(self.myID, data = [self.photon])
		else:
			raise Exception("no side specified")
		self.num_electron_attempts += 1
		self.apply_noise_on_nuclear()
	
	def set_other_conn(self, conn, test_conn):
		# Set up a connection to side 2.
		self.other_conn = conn
		self.test_other_conn = test_conn
	
	def set_repeater_conn(self, conn_to_repeater_control):
		# Set up a connection to the repeater control which will collect entanglement information
		# from the atoms.
		# The connection should be a ClassicalConnection.
		self.repeater_conn = conn_to_repeater_control

	def start_BK(self, args = None):
		# Called when the repeater register should do Barrett-Kok with adjacent registers.
		# Check the message.
		# 	side = 1: self.conn
		# 	side = 2: self.other_conn
		# Also called when the repeater register should do a BSM, or when the repeater register
		# should move the qubit from the electron spin to the nuclear spin.
		data, _ = self.repeater_conn.get_as(self.myID)
		for info in data:
			msg = info[0] # info is sent as a tuple
			if self.to_print: print(self.myID, info)
			if msg == 'start':
				side = info[1]
				# Check if we are starting a BK protocol from scratch, and count num_electron_attempts.
				if side != self.side:
					self.num_electron_attempts = 0
				if side == 1 or side == 2: self.side = side
				# Start the Barrett-Kok protocol by sending photons down to the beamsplitter.
				self.run_protocol_inner()
			elif msg == 'BSM':
				BSMoutcome = BSM(self.nodememory.get_qubit(0), self.nodememory.get_qubit(1),\
									self.gate_fidelity, self.meas_fidelity)
				# self.num_electron_attempts here was reset when the second side started.
				# In particular, we made the nuclear spin active right before starting the second side.
				self.repeater_conn.put_from(self.myID, data = \
									[('BSM_success', self.num_electron_attempts, \
												(BSMoutcome[0][0][0], BSMoutcome[0][1][0])),])
				self.side = 0
				self.stage = 0
				self.nodememory.release_qubit(0)
				self.nodememory.release_qubit(1)
			elif msg == 'move_qubit': # i.e. move from electron spin to nuclear spin
				self.move_qubit(0, 1)
				# Model noise as depolarization channel.
				self.gate_depol([self.nodememory.get_qubit(1)])
				# Let repeater control know.
				self.repeater_conn.put_from(self.myID, data = [('move_qubit_success', self.side),])

	def verification(self, side = None, args = None):
		to_print = False # separate from to_print, since we only want to print this for debugging
		if side is None: 
			super().verification(args = None)
		else:
			# Barrett-Kok has not been completed, so self.side should not have changed yet.
			# Callback function when the first phase of Barrett-Kok is successful.
			if side == 1:
				data, _ = self.test_conn.get_as(self.myID)
			elif side == 2:
				data, _ = self.test_other_conn.get_as(self.myID)
			if self.to_print: print(self.myID, data)
			assert side == self.side, str(self.myID)+', '+str(side)+', '+str(self.side)
			[msg, ] = data
			if msg[:7] == 'success' and self.stage == 1:
				# The incoming message is structured as 'success01' or 'success10'.
				self.detector_order = msg[-2:]
				self.stage = 2
			elif msg[:7] == 'success' and self.stage == 2:
				self.stage = 3
				# At this point, the two atoms are entangled, up to a phase.
				# Make the phase correction if different detectors observed the single photon,
				# because that would cause a pi phase difference.
				if self.to_correct[side-1] and msg[-2:] != self.detector_order:
					# Each node only needs to do pi/2.
					self.nodememory.operate(ns.Z, 0)
				self.detector_order = None
				# When BK is performed, both registers will print states. Only use the second printed 
				# state, because the first state may be printed before the second phase change was made.
				if to_print: print('node', self.myID, '\n', self.atom.qstate.dm)
				if self.repeater_conn is not None:		
					# If success, inform the repeater control.
					# Also pass along which side was successful.
					self.repeater_conn.put_from(self.myID, data = [("success", self.side),])
			else:
				if self.stage == 2: pass # can choose to print here, or at every failure
				if to_print: print('node', self.myID, '\n', self.atom.qstate.dm)
				self.stage = 0
				self.num_sent_photons = 0
	
	def move_qubit(self, index_from, index_to, args = None):
		# Moves qubit from index_from to index_to in the quantum memory of self.node.
		self.nodememory.add_qubit(self.nodememory.get_qubit(index_from), index_to)
		self.nodememory.release_qubit(index_from)

	def run_protocol(self):
		pass

