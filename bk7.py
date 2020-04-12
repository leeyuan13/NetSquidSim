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

from easysquid.quantumMemoryDevice import QuantumMemoryDevice, UniformStandardMemoryDevice
from netsquid.components import QuantumNoiseModel

import logging

class AtomProtocol(TimedProtocol):
	''' Protocol for atom. '''

	def __init__(self, time_step, node, connection, test_conn, to_run = True, start_time = 0,\
					to_correct = False, to_print = False):
		super().__init__(time_step, start_time = start_time, node = node, connection = connection)
		# node = QuantumNode for atom that runs this AtomProtocol
		# node should have a QuantumMemoryDevice at node.qmem
		# connection = QuantumConnection for atom to 50:50 beam splitter
		# test_conn = ClassicalConnection from 50:50 beam splitter to atom

		# Keeping track of nodes.
		self.myID = self.node.nodeID
		if self.conn.idA == self.myID:
			self.otherID = self.conn.idB
		elif self.conn.idB == self.myID:
			self.otherID = self.conn.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		# Quantum memory storage.
		self.nodememory = self.node.qmem

		# Atom and photon state.
		self.atom = None
		self.photon = None
		
		# Connection through which instructions from the beam splitter will be delivered.
		self.test_conn = test_conn

		# Stage of Barrett-Kok protocol.
		# 0 = idle
		# 1 = sent out first photon, waiting for "failed" or "next stage" instruction
		# 2 = sent out second photon, waiting for "failed" or "succeeded" signal
		# 3 = succeeded! (if "failed", then state 0)
		self.stage = 0

		# Whether to initiate entanglement protocol at each time step.
		self.to_run = to_run
		
		# Useful operator.
		self.permute34 = nq.Operator("permute34",\
					 np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))

		# Keep track of which detector observed the photon, so that phase corrections can be made.
		self.detector_order = None
		# Whether this node should correct for phase differences.
		# In BK, only one node needs to make the correction to eliminate phase ambiguity.
		# If we don't care about phase ambiguity, both nodes can have to_correct = False.
		self.to_correct = to_correct

		# Whether to print internal procedures to console.
		self.to_print = to_print

	def set_state(self):
		# Atom to be in [1, 1] (to be normalized).
		# Atomic state [0, 1] is the state resonant with the laser transition.
		# Apply entanglement with photon to get [1, 0] x [1, 0] + [0, 1] x [0, 1].
		# (The photon state is represented in the Fock basis.)
		self.atom, self.photon = nq.create_qubits(2)
		phiP = nq.Operator("phiP", \
				np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]) / np.sqrt(2))
		nq.operate([self.atom, self.photon], phiP)

		# Now, atom and photon are entangled states.
		# Set quantum memory to the atom state.
		self.nodememory.add_qubit(self.atom, 0)

	def send_photon(self):
		# Send photon down the connection.
		self.conn.put_from(self.myID, data = [self.photon])

	def run_protocol_inner(self):
		# Reset self.atom and self.photon.
		self.stage = 0
		self.atom, self.photon = None, None
		self.nodememory.release_qubit(0)
		self.set_state()
		self.send_photon()
		self.stage = 1
		if self.to_print: print("{} sent!".format(self.myID))

	def run_protocol(self):
		if self.to_run:
			self.run_protocol_inner()

	def verify_state(self):
		# Flip atomic state.
		# Recall that ns.X is the permutation operator.
		self.nodememory.operate(ns.X, 0) # note that self.atom is mutated too

		# Create new photonic Fock state.
		# (The old self.photon was already measured.)
		self.photon, = nq.create_qubits(1)
		
		# Entangle self.atom and self.photon.
		# Note that self.photon is initially [1, 0], i.e. in state |0>.
		nq.operate([self.atom, self.photon], self.permute34)
		
		self.send_photon()

		if self.to_print: print("verification {} sent!".format(self.myID))

	def verification(self, args = None):
		# Callback function when the first phase of Barrett-Kok is successful.
		data, _ = self.test_conn.get_as(self.myID)
		[msg, ] = data
		to_print = False
		if msg[:7] == 'success' and self.stage == 1:
			self.verify_state()
			# The incoming message is structured as 'success01' or 'success10'.
			self.detector_order = msg[-2:]
			self.stage = 2
		elif msg[:7] == 'success' and self.stage == 2:
			self.stage = 3
			# At this point, the two atoms are entangled, up to a phase.
			# Make the phase correction if different detectors observed the single photon,
			# because that would cause a pi phase difference.
			if self.to_correct and msg[-2:] != self.detector_order:
				# Each node only needs to do pi/2.
				self.nodememory.operate(ns.Z, 0)
			self.detector_order = None
			# When BK is performed, both nodes will print states. Only use the second printed 
			# state, because the first state may be printed before the second phase change was made.
			if to_print: print('node', self.myID, '\n', self.atom.qstate.dm)
		else:
			if self.stage == 2: pass # can choose to print here, or at every failure
			if to_print: print('node', self.myID, '\n', self.atom.qstate.dm)
			self.stage = 0

	def data_received(self, args = None):
		super().data_received(args)

class BeamSplitterProtocol(TimedProtocol):
	''' Protocol for beam splitter. '''
	def __init__(self, time_step, node, inbound, outbound):
		super().__init__(time_step, node = node, connection = None)
		# time_step = width of time interval in which simultaneous events must fall
		# node = QuantumNode for beam splitter (does not store qubits)

		# Note that inbound and outbound are lists of two QuantumConnection objects each.
		# inbound has connections from atoms to 50:50 beam splitter;
		# outbound has connections from 50:50 beam splitter to detector.

		self.in1, self.in2 = inbound
		self.out1, self.out2 = outbound

		# Keeping track of nodes.
		self.myID = self.node.nodeID
		self.otherIn = [None, None]
		self.otherOut = [None, None]

		if self.in1.idA == self.myID:
			self.otherIn[0] = self.in1.idB
		elif self.in1.idB == self.myID:
			self.otherIn[0] = self.in1.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.in2.idA == self.myID:
			self.otherIn[1] = self.in2.idB
		elif self.in2.idB == self.myID:
			self.otherIn[1] = self.in2.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.out1.idA == self.myID:
			self.otherOut[0] == self.out1.idB
		elif self.out1.idB == self.myID:
			self.otherOut[0] == self.out1.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.out2.idA == self.myID:
			self.otherOut[1] == self.out2.idB
		elif self.out2.idB == self.myID:
			self.otherOut[1] == self.out2.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		# Record details about the last incoming photon that has not been paired (else None).
		# [channel, qubit, time]
		self.last_incoming = None
	
	def incoming(self, channel, args = None):
		# Method to run when qubits are sent along the incoming connections.
		# For now, only pass qubits through the beam splitter if there are Fock states in
		# both channels.
		# If there is no Fock state in one of the channels, the simulation failed to set the
		# state of one of the atomic nodes correctly.
		# channel = 1 or 2, depending on which incoming Connection was used.

		# Beam splitter transformation.
		Umat = np.array([[1, 0, 0, 0],
						 [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
						 [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
						 [0, 0, 0, 1]])
		BSOp = nq.Operator("BSOp", Umat)

		#print("incoming args:", args)
		in_conn = self.in1 if channel == 1 else self.in2 if channel == 2 else None

		[data,], _ = in_conn.get_as(self.myID)
		if self.last_incoming is None:
			self.last_incoming = [channel, data, ns.simutil.sim_time()]
		# Check if the current qubit has been sent at a time close to the previous qubit.
		# Events that occur <= time_step (in ns) apart are considered simultaneous.
		elif ns.simutil.sim_time() - self.last_incoming[-1] <= self.timeStep \
			and channel != self.last_incoming[0]: 
			# Operate directly on photonic Fock state qubits.
			# TODO: is this correct? Should entangle atomic qubits.
			if channel == 2:
				nq.operate([self.last_incoming[1], data], BSOp)
				self.outgoing([self.last_incoming[1], data])
			else:
				nq.operate([data, self.last_incoming[1]], BSOp)
				self.outgoing([data, self.last_incoming[1]])
			self.last_incoming = None
		else:
			self.last_incoming = [channel, data, ns.simutil.sim_time()]

	def incoming1(self, args = None):
		return self.incoming(1, args)
	def incoming2(self, args = None):
		return self.incoming(2, args)

	def outgoing(self, out_data):
		self.out1.put_from(self.myID, data = [out_data[0]])
		self.out2.put_from(self.myID, data = [out_data[1]])
		#print("sent out")

	def data_received(self, args = None):
		# Will not be run if the handler is not registered!!

		#print("args:", args)
		super().data_received(args)
	
	def run_protocol(self):
		pass

class DetectorProtocol(TimedProtocol):
	''' Protocol for detectors. '''
	def __init__(self, time_step, node, connection, test_conn = None, to_print = False, \
					pdark = 1e-7, efficiency = 0.93):
		super().__init__(time_step, node = node, connection = connection)
		# node = QuantumNode for detector (does not store qubits)
		# test_conn = ClassicalConnection back to beamsplitter 

		self.test_conn = test_conn
		self.to_print = to_print
		# Dark count probability.
		# pdark = prob of measuring 1 when Fock state is 0.
		# Default value of pdark = 1e-7 comes from ~100 dark counts per second, 
		# and 12 ns (= dead time) windows.
		self.pdark = pdark
		# Detector efficiency.
		self.efficiency = efficiency

		# Keeping track of nodes.
		self.myID = self.node.nodeID
		if self.conn.idA == self.myID:
			self.otherID = self.conn.idB
		elif self.conn.idB == self.myID:
			self.otherID = self.conn.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

	def process_data(self):
		[data,], _ = self.conn.get_as(self.myID)
		
		# Model losses due to finite detection efficiency using amplitude damping.
		nq.qubitapi.amplitude_dampen(data, 1.-self.efficiency, prob = 1)

		# Model dark counts using (reverse) amplitude damping.
		# Due to entanglement, these dark counts will affect the statistics of the
		# atoms' density matrices.
		nq.qubitapi.amplitude_dampen(data, self.pdark, prob = 0)

		success, prob = nq.measure(data, observable = ns.Z)
		# Note ns.Z = [[1, 0], [0, -1]].
		# success == 0 means that the positive eigenvalue was observed;
		# success == 1 means that the negative eigenvalue was observed instead.
		if self.to_print: print("{} measured {} with probability {}".format(self.myID, success, prob))

		if self.test_conn is not None:
			self.test_conn.put_from(self.myID, data = [success])

class StateCheckProtocol(TimedProtocol):
	''' 
	Protocol for beamsplitter, to check the state of atomic qubits. 
	Written separately from BeamSplitterProtocol, since we can treat BeamSplitterProtocol
	as solely a protocol for 2-channel unitary transformation and message-passing.
	'''

	def __init__(self, time_step, node, from_detectors, to_atoms):
		super().__init__(time_step, node = node, connection = None)
		# time_step = width of time interval in which simultaneous events must fall
		# node = QuantumNode for beam splitter (does not store qubits)

		# Note that from_detectors and to_atoms are lists of two QuantumConnection objects each.
		# inbound has connections from atoms to 50:50 beam splitter;
		# outbound has connections from 50:50 beam splitter to detector.

		self.in1, self.in2 = from_detectors
		self.out1, self.out2 = to_atoms

		# Keeping track of nodes.
		self.myID = self.node.nodeID
		self.otherIn = [None, None]
		self.otherOut = [None, None]

		if self.in1.idA == self.myID:
			self.otherIn[0] = self.in1.idB
		elif self.in1.idB == self.myID:
			self.otherIn[0] = self.in1.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.in2.idA == self.myID:
			self.otherIn[1] = self.in2.idB
		elif self.in2.idB == self.myID:
			self.otherIn[1] = self.in2.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.out1.idA == self.myID:
			self.otherOut[0] == self.out1.idB
		elif self.out1.idB == self.myID:
			self.otherOut[0] == self.out1.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")

		if self.out2.idA == self.myID:
			self.otherOut[1] == self.out2.idB
		elif self.out2.idB == self.myID:
			self.otherOut[1] == self.out2.idA
		else:
			raise EasySquidException("Attempt to run protocol at remote nodes")
		
		# Record details about the last incoming message that has not been paired (else None).
		# [channel, message, time]
		self.last_incoming = None
	
	def incoming(self, channel, args = None):
		# Method to run when messages are sent along the incoming connections.
		in_conn = self.in1 if channel == 1 else self.in2 if channel == 2 else None
		[data], _ = in_conn.get_as(self.myID)
		if self.last_incoming is None:
			self.last_incoming = [channel, data, ns.simutil.sim_time()]
		# Check if the current qubit has been sent at a time close to the previous qubit.
		# For now, events that occur <= 1 ns apart are considered simultaneous.
		elif ns.simutil.sim_time() - self.last_incoming[-1] <= self.timeStep \
			and channel != self.last_incoming[0]:
			# Check if the detectors, combined, only detected one photon.
			# Also send data on which detector measured a photon, so that the atoms can
			# correct for the phase difference.
			# "success01" means out1 measured 0 and out2 measured 1
			# "success10" means the opposite.
			
			# Construct detector order.
			if channel == 1: suffix = str(int(data)) + str(int(self.last_incoming[1]))
			else: suffix = str(int(self.last_incoming[1])) + str(int(data))

			# Send messages.
			if data + self.last_incoming[1] == 1:
				self.outgoing("success" + suffix)
			else:
				self.outgoing("failure" + suffix)
			self.last_incoming = None
		else:
			self.last_incoming = [channel, data, ns.simutil.sim_time()]

	def incoming1(self, args = None):
		return self.incoming(1, args)
	def incoming2(self, args = None):
		return self.incoming(2, args)

	def outgoing(self, out_data):
		self.out1.put_from(self.myID, data = [out_data])
		self.out2.put_from(self.myID, data = [out_data])
		#print("sent out")

class QubitLossNoiseModel(QuantumNoiseModel):
	''' 
	Account for lossy fibres using noise on Fock states. 
	Apparently consistent with T1, T2 noise, ignoring the "times".
	'''
	def __init__(self, prob_loss):
		# prob_loss = probability of losing the photon
		self.prob_loss = prob_loss
	
	def noise_operation(self, qubits, delta_time = 0):
		for qubit in qubits:
			self.apply_noise(qubit)
	
	def apply_noise(self, qubit):
		# Applies noise [0, 1] -> [1, 0] to the qubit.
		# prob = 1 below means that [1, 0] cannot go up to [0, 1].
		nq.qubitapi.amplitude_dampen(qubit, self.prob_loss, prob = 1)

if __name__ == '__main__':
	ns.simutil.sim_reset()
	nq.set_qstate_formalism(ns.QFormalism.DM)

	# Atoms are modeled as QuantumMemoryDevice objects.
	# Add noise models as desired.
	atom1 = QuantumMemoryDevice("atom1", 1)
	atom2 = QuantumMemoryDevice("atom2", 2)

	#atom1 = UniformStandardMemoryDevice("atom1", 1, T1 = 10, T2 = 5)
	#atom2 = UniformStandardMemoryDevice("atom2", 1, T1 = 10, T2 = 5)

	source1 = QuantumNode("source1", 1, memDevice = atom1)
	source2 = QuantumNode("source2", 2, memDevice = atom2)

	detector1 = QuantumNode("detector1", 4)
	detector2 = QuantumNode("detector2", 5)

	beamsplitter = QuantumNode("beamsplitter", 3)

	# To incorporate losses, need to define new FibreLossModel objects for each Connection.
	conn1 = QuantumConnection(source1, beamsplitter, noise_model = QubitLossNoiseModel(0.1))
	conn2 = QuantumConnection(source2, beamsplitter, noise_model = QubitLossNoiseModel(0.1))
	
	# Link the detectors back to the beam splitter.
	test_conn1 = ClassicalConnection(detector1, beamsplitter)
	test_conn2 = ClassicalConnection(detector2, beamsplitter)
	test_conn3 = ClassicalConnection(beamsplitter, source1)
	test_conn4 = ClassicalConnection(beamsplitter, source2)

	# Need to place this after the test connections, so that we can reference them.
	conn3 = QuantumConnection(beamsplitter, detector1)
	conn4 = QuantumConnection(beamsplitter, detector2)
	
	proto1 = AtomProtocol(50, source1, conn1, test_conn3, to_correct = True)
	proto2 = AtomProtocol(50, source2, conn2, test_conn4)

	proto3 = DetectorProtocol(50, detector1, conn3, test_conn1)
	proto4 = DetectorProtocol(50, detector2, conn4, test_conn2)

	BSproto = BeamSplitterProtocol(0.01, beamsplitter, [conn1, conn2], [conn3, conn4])
	SCproto = StateCheckProtocol(0.01, beamsplitter, [test_conn1, test_conn2], [test_conn3, test_conn4])

	source1.setup_connection(conn1, [proto1])
	source2.setup_connection(conn2, [proto2])
	test_conn3.register_handler(source1.nodeID, proto1.verification)
	test_conn4.register_handler(source2.nodeID, proto2.verification)

	# Set up protocol without using QuantumNode.setup_connection
	# since the whole setup assumes only one connection per protocol.
	# Instead, register handlers manually.
	# No need to define nodes and connections for the protocol, since they are
	# set above.
	# Note that BSproto.data_received will not be run because that requires us
	# to use beamsplitter.setup_connection.
	conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
	conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
	test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
	test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)

	detector1.setup_connection(conn3, [proto3])
	detector2.setup_connection(conn4, [proto4])

	logger.setLevel(logging.DEBUG)

	# Need to start nodes with periodic behavior.
	source1.start()
	source2.start()
	beamsplitter.start()

	ns.simutil.sim_run(250)
