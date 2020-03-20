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
from netsquid.components import QuantumNoiseModel

import logging

from bk6 import AtomProtocol, BeamSplitterProtocol, DetectorProtocol,\
				StateCheckProtocol, QubitLossNoiseModel
# Use BSM2 because it encodes the BSM result in a way that tells us what gates are needed for
# teleportation to actually occur.
from BSM2 import BellStateMeasurement as BSM

np.set_printoptions(precision=2)

# Multiple channels from Alice to Bob through repeater node, arranged in hybrid fashion.
# A---N---C1--(fully bipartite)--C2---N---B
# This time, packaged nicely to only expose Alice's and Bob's scont nodes.
# Also with proper C1, C2 connections i.e. with electron and nuclear spins.

# The repeater connections need not be perfect, and we incorporate the loss model in Rozpedek
# et al for the decoherence when an electron attempt is made.

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
	def __init__(self, time_step, node, connection, test_conn, to_correct = False):
		super().__init__(time_step, node, connection, test_conn, to_run = False, to_correct = to_correct)
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
	def __init__(self, time_step, node, connection, test_conn, other_conn = None, \
							test_other_conn = None, to_correct = False, to_print = False,\
							noise_on_nuclear = None):
		# to_correct tells us whether the protocol should correct for phase in Barrett-Kok.
		# other_conn, test_other_conn are the connections to the other source nodes.
		# noise_on_nuclear is the noise to be applied on the nuclear spin every time the electron spin is
		#	used; it comprises a dephasing and depolarization, see Rozpedek et al.
		super().__init__(time_step, node, connection, test_conn, to_run = False, to_correct = to_correct)
		# Note that the quantum memory has 2 atoms now:
		#	electron spin (active) = index 0
		# 	nuclear spin (storage) = index 1
		self.repeater_conn = None # classical connection to repeater control
		self.other_conn = None # quantum connection to repeater beamsplitter
		self.test_other_conn = None # classical connection to repeater beamsplitter
		self.num_electron_attempts = 0 # number of photons sent by the electron spin before a
									   # successful entanglement is achieved.
									   # This is important because it may affect the nuclear spin.

		# To keep track of which detector observed the photon.
		self.detector_order = None
		self.to_correct = to_correct

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

	# Override self.send_photon(), since either connection could be used.
	def send_photon(self):
		return # do nothing; send photon manually
	
	def set_other_conn(self, conn, test_conn):
		# Set up a connection to side 2.
		self.other_conn = conn
		self.test_other_conn = test_conn
	
	def set_repeater_conn(self, conn_to_repeater_control):
		# Set up a connection to the repeater control which will collect entanglement information
		# from the atoms.
		# The connection should be a ClassicalConnection.
		self.repeater_conn = conn_to_repeater_control

	def start(self, args = None):
		# Called when the repeater node should do Barrett-Kok with the source nodes.
		# Check the message.
		# 	side = 1: self.conn
		# 	side = 2: self.other_conn
		# Also called when the repeater node should do a BSM, or when the repeater node
		# should move the qubit from the electron spin to the nuclear spin.
		[info], _ = self.repeater_conn.get_as(self.myID)
		msg, side = info
		if msg == 'start':
			# Start the Barrett-Kok protocol by sending photons down to the beamsplitter.
			self.run_protocol_inner()
			# Check if we are starting a BK protocol from scratch, and count num_electron_attempts.
			if side != self.side:
				self.num_electron_attempts = 0
			# Send the photon.
			if side == 1:
				self.side = 1
				self.conn.put_from(self.myID, data = [self.photon])
				# Increment self.num_electron_attempts.
				self.num_electron_attempts += 1
				# Apply noise on nuclear spin.
				if self.noise_on_nuclear is not None:
					nuclear_spin = self.nodememory.get_qubit(1)
					if nuclear_spin is not None:
						self.noise_on_nuclear(nuclear_spin)
			elif side == 2:
				self.side = 2
				self.other_conn.put_from(self.myID, data = [self.photon])
				self.num_electron_attempts += 1
				if self.noise_on_nuclear is not None:
					nuclear_spin = self.nodememory.get_qubit(1)
					if nuclear_spin is not None:
						self.noise_on_nuclear(nuclear_spin)
		elif msg == 'BSM':
			BSMoutcome = BSM(self.nodememory.get_qubit(0), self.nodememory.get_qubit(1))
			# self.num_electron_attempts here was reset when the second side started.
			# In particular, we made the nuclear spin active right before starting the second side.
			self.repeater_conn.put_from(self.myID, data = \
								[('BSM_success', self.num_electron_attempts, \
											(BSMoutcome[0][0][0], BSMoutcome[0][1][0])),])
			self.side = 0
			self.stage = 0
		elif msg == 'move_qubit': # i.e. move from electron spin to nuclear spin
			self.move_qubit(0, 1)
			# Let repeater control know.
			self.repeater_conn.put_from(self.myID, data = [('move_qubit_success',)])

	def verification(self, side = None, args = None):
		to_print = False # separate from to_print, since we only want to print this for debugging
		if side is None: 
			super().verification(args = None)
		else:
			assert side == self.side
			# Callback function when the first phase of Barrett-Kok is successful.
			if self.side == 1:
				data, _ = self.test_conn.get_as(self.myID)
			elif self.side == 2:
				data, _ = self.test_other_conn.get_as(self.myID)
			[msg, ] = data
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
		if self.stage == 3 and self.repeater_conn is not None:
			# If success, inform the repeater control.
			# Also pass along which side was successful.
			self.repeater_conn.put_from(self.myID, data = [("success", self.side),])

	# Note: need to redefine verify_state, since send_photon does nothing now.
	def verify_state(self):
		super().verify_state()
		# Send photon.
		if self.side == 1:
			self.conn.put_from(self.myID, data = [self.photon])
		elif self.side == 2:
			self.other_conn.put_from(self.myID, data = [self.photon])
		# Increment num_electron_attempts.
		self.num_electron_attempts += 1
		# Apply noise on nuclear spin.
		if self.noise_on_nuclear is not None:
			nuclear_spin = self.nodememory.get_qubit(1)
			if nuclear_spin is not None:
				self.noise_on_nuclear(nuclear_spin)

	def move_qubit(self, index_from, index_to, args = None):
		# Moves qubit from index_from to index_to in the quantum memory of self.node.
		self.nodememory.add_qubit(self.nodememory.get_qubit(index_from), index_to)
		self.nodememory.release_qubit(index_from)

	def run_protocol(self):
		pass

class RepeaterControlProtocol(TimedProtocol):
	''' Protocol for repeater control. '''
	def __init__(self, time_step, node, rep, conns):
		# conn, rep = list of connections from repeater nodes to the repeater control
		# Need to have access to the repeater nodes so that the Bell state measurement can be performed.
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.conns = conns
		self.rep = rep

		# Connection to sconts.
		self.scont_conn = [None, None]

		# Lists of source nodes on either side.
		# These are connected to self.rep.
		# Could be useful for data collection.
		self.source1 = None
		self.source2 = None
		
		# Status = 1 if the repeater node is entangled with Alice/Bob
		self.status1 = [0]*len(rep)
		self.status2 = [0]*len(rep)

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

	def is_entangled(self, index, args = None):
		# Method to run when the repeater nodes inform the control that entanglement has been
		# established, either with Alice or with Bob.
		# The index can be determined by setting up the callback-channel linkage appropriately.
		# The side is determined by the message that the repeater node sends.
		[data], _ = self.conns[index].get_as(self.myID)
		if data[0] == "success": # Barrett-Kok success
			if data[1] == 1: # data[1] = side
				if self.to_collect_data: self.times1.append(sim_time())
				# Ask the repeater node to move qubits.
				# Data down the ClassicalConnection is always a two-element tuple.
				self.conns[index].put_from(self.myID, data = [("move_qubit", None),])
			elif data[1] == 2:
				self.status2[index] = 1
				if self.to_collect_data:
					self.times2.append(sim_time())
				# Ask qubits to perform BSM.
				self.conns[index].put_from(self.myID, data = [("BSM", None),])
		elif data[0] == "move_qubit_success": # successfully moved qubit
			# Set the status of the repeater node so that it can start sending photons to Bob.
			self.status1[index] = 1
			# The repeater control sends the BK start signal under run_protocol, so it
			# does it periodically.
		elif data[0] == "BSM_success": # successfully performed BSM
			num_electron_attempts = data[1]
			BSM_outcome = data[2]
			# Perform required unitaries.
			# Only need to perform these unitaries on one side, e.g. Alice.
			if self.source1 is not None and self.source2 is not None:
				if BSM_outcome[0] == 1:
					nq.operate(self.source1[index].qmemory.get_qubit(0), ns.X)
				if BSM_outcome[1] == 1:
					nq.operate(self.source1[index].qmemory.get_qubit(0), ns.Z)
			# Restore statuses to normal, i.e. set status back to zero.
			self.status1[index] = 0
			self.status2[index] = 0
			# Collect data.
			if self.to_collect_data:
				# No need to keep track of when BSM started, since BSM will always "succeed".
				self.key_times.append((sim_time(), None, num_electron_attempts))
				# Record density matrix, if possible.
				if self.source1 is not None and self.source2 is not None:
					self.key_dm.append(nq.reduced_dm([self.source1[index].qmemory.get_qubit(0),\
									self.source2[index].qmemory.get_qubit(0)]))
				# Note: if multiple messages are sent down the same channel in the
				#  same timestep, then the messages will be concatenated in a list.
				#  Hence, better to group messages in tuples, as multiple BSMs may mean
				#  multiple messages sent at the same time.
				self.scont_conn[0].put_from(self.myID, data = [('bsm', index)])
				self.scont_conn[1].put_from(self.myID, data = [('bsm', index)])
	
	def set_scont_conn(self, conn1, conn2):
		# Set connections to sconts.
		self.scont_conn = [conn1, conn2]

	def set_sources(self, source1, source2):
		# Set the sources that rep1 and rep2 are connected to.
		self.source1, self.source2 = source1, source2

	def run_protocol(self):
		# Notify source controls about the successful entanglements.
		# Note that messages sent down the scont-to-repeater control connection must be
		# bundled in tuples, because the same connection can be used to send multiple BSM
		# results simultaneously.
		to_BK1 = [1-x for x in self.status1]
		to_BK2 = [self.status1[i] * (1-self.status2[i]) for i in range(len(self.status1))]
		self.scont_conn[0].put_from(self.myID, data = [('status update', to_BK1),])
		self.scont_conn[1].put_from(self.myID, data = [('status update', to_BK2),])
		# Notify repeater nodes that need to start BK.
		for i in range(len(self.status1)):
			if to_BK1[i] == 1: # remember, NOT "if i == 0"
				self.conns[i].put_from(self.myID, data = [('start', 1),])
			if to_BK2[i] == 1:
				self.conns[i].put_from(self.myID, data = [('start', 2),])
	
class ScontProtocol(TimedProtocol):
	''' Protocol for source control.'''
	def __init__(self, time_step, node, sources, conns, to_print = False):
		super().__init__(time_step, node = node, connection = None)

		self.myID = self.node.nodeID
	
		self.sources = sources
		self.conns = conns

		# Connection to repeater control.
		self.rep_conn = None

		# to_BK = 1 if the source node is not yet entangled and should perform BK
		self.to_BK = [0]*len(conns)

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
				self.to_BK = data
				# Get source nodes to start Barrett-Kok.
				for i in range(len(self.conns)):
					if self.to_BK[i] == 1:
						self.conns[i].put_from(self.myID, data = ['start'])
					# If the source node is waiting for entanglement on the other side,
					# don't do anything.
			elif msg == 'bsm':
				if self.to_print:
					print('bsm', self.myID, data, sim_time())
					print(self.sources[data].qmemory.get_qubit(0).qstate)
				pass # don't need to make measurements for now

# Functions needed for runtime behavior.
if True:
	def connect_by_trad_repeater(num_channels, scont1, scont2, repeater_control, \
							make_BK, make_scont_conn, make_rep_control_conn):
		# Connects the source controls scont1, scont2 via the repeater control rep_control.
		# Inputs:
		#	num_channels = (int) number of channels connected in the traditional/QuTech fashion
		#	scont1, scont2, rep_control = QuantumNodes
		# 	make_BK = function that returns [[sender_node, rep_node, other_sender_node], 
		#									 [sender_prot, rep_prot, other_sender_prot]]
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
		#	make_rep_control_conn = function that takes (rep, rep_proto, 
		#							 rep_control) and connects repeater nodes to repeater controls, 
		#							 returning [rep_control_prot, conns]
		#							 Note that the repeater control protocol is NOT purely responsive,
		#							 as it has to initiate the Barrett-Kok sequence at every specified
		#							 time interval. Therefore, connect_to_rep_control should add
		#							 control_prot as a node protocol.
		# Returns:
		#	rep_control_prot, scont1_prot, scont2_prot

		# Construct source nodes and repeater nodes (which actually contain atoms).
		source1 = [] # Alice
		source1_proto = []
		rep = []
		rep_proto = [] # repeater protocols
		source2 = [] # Bob
		source2_proto = []

		for i in range(num_channels):
			[a, b, other_a], [c, d, other_c] = make_BK() 
			source1.append(a) # source node
			rep.append(b) # repeater node
			source1_proto.append(c)
			rep_proto.append(d) # repeater protocol
			source2.append(other_a) # other source node
			source2_proto.append(other_c)
			
		scont1_prot, _ = make_scont_conn(source1, source1_proto, scont1)
		scont2_prot, _ = make_scont_conn(source2, source2_proto, scont2)

		control_prot, _ = make_rep_control_conn(rep, rep_proto, repeater_control) 

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

		# If the repeater_bs connections are not made in this functionk, we also need to return 
		# repeater nodes and protocols; necessary for establishing repeater beamsplitters
		# later.	
		# Since we are making repeater_bs_connections here, this is not necessary.
		return control_prot, scont1_prot, scont2_prot

	def create_BK(index_gen, make_atom_memory, make_rep_memory, node_to_bs, bs_to_det, make_atom_prot, \
					make_rep_prot, make_det_prot, make_bs_prot):
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
		# Returns:
		#	[sender_node, rep_node], [sender_prot, rep_prot]

		# Source, i.e. Alice.
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
		test_conn3 = ClassicalConnection(beamsplitter, sender_node)
		test_conn4 = ClassicalConnection(beamsplitter, rep_node)
		# Set up protocols at nodes, detectors and beamsplitter.
		# Note that the atom/repeater protocol must have a "verification" method as the callback.
		proto1 = make_atom_prot(sender_node, conn1, test_conn3, to_correct = True)
		proto2 = make_rep_prot(rep_node, conn2, test_conn4)
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
		test_conn4.register_handler(rep_node.nodeID, \
								lambda args = None: proto2.verification(side = 1, args = args))
		conn1.register_handler(beamsplitter.nodeID, BSproto.incoming1)
		conn2.register_handler(beamsplitter.nodeID, BSproto.incoming2)
		test_conn1.register_handler(beamsplitter.nodeID, SCproto.incoming1)
		test_conn2.register_handler(beamsplitter.nodeID, SCproto.incoming2)
		detector1.setup_connection(conn3, [proto3])
		detector2.setup_connection(conn4, [proto4])
		# Note that the repeater protocol need not be added as a node protocol for the
		# repeater node: see run_protocol in e.g. BellProtocol.
		
		# Add other source, i.e. Bob.
		index = index_gen.__next__()
		other_sender_atom = make_atom_memory("atom"+str(index), 1)
		other_sender_node = QuantumNode("source"+str(index), index, memDevice = other_sender_atom)
		# Detectors.
		index = index_gen.__next__()
		other_detector1 = QuantumNode("detector"+str(index), index)
		index = index_gen.__next__()
		other_detector2 = QuantumNode("detector"+str(index), index)
		# Beamsplitter.
		index = index_gen.__next__()
		other_beamsplitter = QuantumNode("beamsplitter"+str(index), index)
		# Quantum connections from node to beamsplitter.
		other_conn1 = node_to_bs(other_sender_node, other_beamsplitter)
		other_conn2 = node_to_bs(rep_node, other_beamsplitter)
		# Quantum connections from beamsplitter to detectors.
		other_conn3 = bs_to_det(other_beamsplitter, other_detector1)
		other_conn4 = bs_to_det(other_beamsplitter, other_detector2)
		# Classical connections from detectors to beamsplitter to nodes.
		other_test_conn1 = ClassicalConnection(other_detector1, other_beamsplitter)
		other_test_conn2 = ClassicalConnection(other_detector2, other_beamsplitter)
		other_test_conn3 = ClassicalConnection(other_beamsplitter, other_sender_node)
		other_test_conn4 = ClassicalConnection(other_beamsplitter, rep_node)
		# Set up protocols at nodes, detectors and beamsplitter.
		# Note that the atom/repeater protocol must have a "verification" method as the callback.
		other_proto1 = make_atom_prot(other_sender_node, other_conn1, other_test_conn3, \
							to_correct = True)
		proto2.set_other_conn(other_conn2, other_test_conn4) 
		# proto2 is for the repeater node, so no "other_proto2".
		other_proto3 = make_det_prot(other_detector1, other_conn3, other_test_conn1)
		other_proto4 = make_det_prot(other_detector2, other_conn4, other_test_conn2)
		other_BSproto = make_bs_prot[0](other_beamsplitter, [other_conn1, other_conn2], \
									[other_conn3, other_conn4])
		other_SCproto = make_bs_prot[1](other_beamsplitter, [other_test_conn1, other_test_conn2], \
									[other_test_conn3, other_test_conn4])
		# Set up handlers.
		# setup_connection does things automatically, but we can also register handlers manually
		# especially for multi-connection nodes.
		other_sender_node.setup_connection(other_conn1, [other_proto1])
		other_test_conn3.register_handler(other_sender_node.nodeID, other_proto1.verification)
		other_test_conn4.register_handler(rep_node.nodeID, \
								lambda args = None: proto2.verification(side = 2, args = args))
		other_conn1.register_handler(other_beamsplitter.nodeID, other_BSproto.incoming1)
		other_conn2.register_handler(other_beamsplitter.nodeID, other_BSproto.incoming2)
		other_test_conn1.register_handler(other_beamsplitter.nodeID, other_SCproto.incoming1)
		other_test_conn2.register_handler(other_beamsplitter.nodeID, other_SCproto.incoming2)
		other_detector1.setup_connection(other_conn3, [other_proto3])
		other_detector2.setup_connection(other_conn4, [other_proto4])
		# Return nodes and protocols.
		# Keep the beamsplitter + detectors hidden away in the abstraction.
		return [sender_node, rep_node, other_sender_node], [proto1, proto2, other_proto1]

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

	def connect_to_repeater(rep, proto, rep_control, make_control_prot):
		# Connects all repeater nodes to a central control, as in the hybrid repeater architecture.
		# Inputs:
		#	rep = list of repeater nodes
		#	proto = list of repeater protocols (same length as rep)
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
		assert len(rep) == len(proto)
		# List of connections between repeater nodes and repeater control.
		conns = []
		for i in range(len(rep)):
			conn = ClassicalConnection(rep[i], rep_control)
			proto[i].set_repeater_conn(conn)
			conns.append(conn)
		# Set up repeater control protocol.
		control_prot = make_control_prot(rep_control, rep, conns)
		# Set up handlers.
		for i in range(len(conns)):
			# Note that having default arguments here is necessary: see
			# https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop. 
			func = lambda args, index = i: control_prot.is_entangled(index, args)
			conns[i].register_handler(rep[i].nodeID, proto[i].start)
			conns[i].register_handler(rep_control.nodeID, func)
		rep_control.add_node_protocol(control_prot)
		return control_prot, conns

# Wrap procedure in a function.
def run_simulation(num_channels, atom_times, rep_times, channel_loss, duration = 100,\
					repeater_channel_loss = None, noise_on_nuclear_params = None):
	# Note that repeater_channel_loss is irrelevant here, because there are no channels
	# within the repeater.
	
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
	BS_CLOCK_RATE = 0.01 # For BeamSplitterProtocol and StateCheckProtocol, for whom time
						 # is supposedly of the essence because they require incoming photons
						 # to be coincient. ('supposedly' because these protocols
						 # do not actually run unprompted, i.e. run_protocol is not run.)
	REPEATER_CONTROL_RATE = 1 # How often the repeater control sends start signals to repeater nodes
							  # that are eligible. (For RepeaterControlProtocol, where run_protocol
							  # is run.)
	
	# Noise on nuclear spin when electron spin sends a photon.
	if noise_on_nuclear_params is None:
		nna, nnb = 0, 0
	else:
		nna, nnb = noise_on_nuclear_params
	noise_on_nuclear = lambda q: nq.qubitapi.multi_operate(q, [ns.I, ns.X, ns.Y, ns.Z], \
														[1-nna-0.75*nnb, 0.25*nnb, 0.25*nnb, (nna+0.25*nnb)])

	#make_atom_memory = lambda x, y: StandardMemoryDevice(x, y, decoherenceTimes = [[211400, 211400],])
	#make_rep_memory = lambda x, y: StandardMemoryDevice(x, y, \
	#										decoherenceTimes = [[211400, 211400], [0, 0]])
	make_atom_memory = lambda x, y: StandardMemoryDevice(x, y, decoherenceTimes = atom_times)
	make_rep_memory = lambda x, y: StandardMemoryDevice(x, y, decoherenceTimes = rep_times)
	node_to_bs = lambda x, y: QuantumConnection(x, y, noise_model = QubitLossNoiseModel(channel_loss))
	bs_to_det = lambda x, y: QuantumConnection(x, y)
	# Note that to_correct is for the phase difference in BK.
	make_atom_prot = lambda x, y, z, to_correct = False: \
							SourceProtocol(BASE_CLOCK_RATE, x, y, z, to_correct=to_correct)
	make_rep_prot = lambda x, y, z, to_correct = False: \
							BellProtocol(BASE_CLOCK_RATE, x, y, z, to_correct=to_correct, \
										noise_on_nuclear = noise_on_nuclear)
	make_det_prot = lambda x, y, z: DetectorProtocol(BASE_CLOCK_RATE, x, y, z)
	make_bs_prot = [lambda x, y, z: BeamSplitterProtocol(BS_CLOCK_RATE, x, y, z),\
					lambda x, y, z: StateCheckProtocol(BS_CLOCK_RATE, x, y, z)]

	make_BK = lambda index_generator = None: \
				create_BK(index_gen if index_generator is None else index_generator, make_atom_memory, \
										make_rep_memory, node_to_bs, bs_to_det, make_atom_prot, \
										make_rep_prot, make_det_prot, make_bs_prot)

	make_scont_prot = lambda x, y, z: ScontProtocol(BASE_CLOCK_RATE, x, y, z)
	make_scont_conn = lambda x, y, z: connect_to_source(x, y, z, make_scont_prot)

	make_control_prot = lambda x, y, z: RepeaterControlProtocol(REPEATER_CONTROL_RATE, \
																		x, y, z)
	make_rep_control_conn = lambda x, y, z: connect_to_repeater(x, y, z, make_control_prot)

	next_index = index_gen.__next__()
	scont1 = QuantumNode("scont"+str(next_index), next_index)
	next_index = index_gen.__next__()
	scont2 = QuantumNode("scont"+str(next_index), next_index)
	next_index = index_gen.__next__()
	repeater_control = QuantumNode("rep_control"+str(next_index), next_index)

	rep_control_prot, scont1_prot, scont2_prot = \
						connect_by_trad_repeater(num_channels, scont1, scont2, \
						repeater_control, make_BK, make_scont_conn, make_rep_control_conn)

	logger.setLevel(logging.DEBUG)

	scont1.start()
	scont2.start()
	repeater_control.start()

	ns.simutil.sim_run(duration=duration)

	return rep_control_prot

if __name__ == '__main__':
	# args: num_channels, atom_times, rep_times, channel_loss
	rep_control_prot = run_simulation(1, [[2, 2],], [[2, 2], [100, 100]], 0.1)
	# No noise:
	#rep_control_prot = run_simulation(1, None, None, 0)
	if False: # plot graphs
		import matplotlib.pyplot as plt
		time1 = rep_control_prot.data[0]
		time2 = rep_control_prot.data[1]
		num_trials = min(len(time1), len(time2))
		wait_time = [abs(time1[i] - time2[i])/REPEATER_CONTROL_RATE for i in range(num_trials)]

		plt.figure()
		plt.hist(wait_time, bins = list(range(int(max(wait_time))+2)))
		plt.xlabel('wait time / clock cycles')
		plt.ylabel('number of occurrences')
		plt.title('Wait time between matches across channels')

		successful_entanglement = [x[0] for x in rep_control_prot.data[2]]
		plt.figure()
		plt.plot(time1, 'b:')
		plt.plot(time2, 'r:')
		plt.plot(successful_entanglement, 'k-')
		plt.xlabel('number of successes')
		plt.ylabel('time of occurrence')

		num_electron_attempts = [x[2] for x in rep_control_prot.data[2]]
		plt.figure()
		plt.hist(num_electron_attempts, bins = list(range(2, int(max(num_electron_attempts))+2)))
		plt.xlabel('number of attempts')
		plt.ylabel('number of occurrences')
		plt.title('Number of attempts at electron entanglement with an active nuclear spin')

		plt.show()
