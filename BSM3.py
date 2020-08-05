import netsquid as ns
import netsquid.qubits as nq
import numpy as np

# NetSquid has ns.CNOT in-built.

def gen_kron(matrices):
	# Generalized Kronecker product for matrices = list of matrices.
	# e.g. if matrices = [X, Y, Z], then gen_kron(matrices) = XYZ (the Pauli operation)
	if len(matrices) == 1:
		return matrices[0]
	else:
		return gen_kron([np.kron(matrices[0], matrices[1]),] + matrices[2:])
	
def depolarization(num_qubits, fidelity):
	# num_qubits = number of qubits in the state
	# fidelity = fidelity of depolarized state
	# The depolarization operation is rho -> fidelity * rho + identity * (1-fidelity)/2**(num_qubits).
	# The Kraus operators for the completely depolarizing channel (fidelity = 0) are just a uniform
	# distribution of general Pauli operators.
	# Returns a depolarization function that you can apply to states.
	# The function takes a list of qubits.

	# Paulis.
	pauli_ns = [ns.I, ns.X, ns.Y, ns.Z]
	paulis = [np.array(p._matrix) for p in pauli_ns]

	# General Paulis.
	indices = lambda i, j: int(i/4**j) % 4 
	# indices(i, *) should give the i-th combination of Paulis in lexicographic order.
	# For instance, if num_qubits = 3, indices(4**3-1, *) = (3, 3, 3) --> (ZZZ)
	# Then indices(i, j) just gives the Pauli on the j-th qubit for the i-th general Pauli.
	gen_paulis = [gen_kron([paulis[indices(i, j)] for j in range(num_qubits)]) \
						for i in range(4**num_qubits)]
	# Get operators.
	gen_pauli_ns = [ns.Operator('pauli'+str(i), gen_paulis[i]) for i in range(4**num_qubits)]
	# Kraus coefficients.
	coeffs = [(1.-fidelity)/4**num_qubits]*(4**num_qubits)
	coeffs[0] += fidelity

	return lambda q: nq.qubitapi.multi_operate(q, gen_pauli_ns, coeffs)
	
def BellStateMeasurement(atom1, atom2, total_gate_fidelity = 1., meas_fidelity = 1.):
	# total_gate_fidelity = cumulative fidelity of two-qubit gates (CNOT, etc.). 
	# meas_fidelity = fidelity of measuring the state of the electron spin.

	# Note that cnot_fidelity should include the effects of e.g. swapping.
	# Let F be the fidelity of a 2-qubit operation.
	# One of [atom1, atom2] is an electron spin; the other is a nuclear spin.
	# Then, we need two 2-qubit gates, giving a cumulative fidelity of F**2.
	# We also need to measure the electron spin twice.

	# Generate depolarization functions.
	gate_depol = depolarization(2, total_gate_fidelity)
	meas_depol = depolarization(1, meas_fidelity)

	# First do a CNOT on 1 and 2.
	nq.operate([atom1, atom2], ns.CNOT)

	# Apply the gate fidelity depolarization.
	gate_depol([atom1, atom2])
	
	# Unclear if the following rotations are necessary?
	# Follow the paper "Unconditional quantum teleportation between distant solid-state quantum bits."

	# Rotate 1 by pi/2 about the y-axis.
	Y2 = nq.create_rotation_op(np.pi/2, (0, 1, 0))
	nq.operate(atom1, Y2)
	# Rotate 2 by pi about the y-axis.
	Y1 = nq.create_rotation_op(np.pi, (0, 1, 0))
	nq.operate(atom2, Y1)
	# Rotate 1 by pi/2 about the y-axis.
	# Seems to be the wrong move??
	#nq.operate(atom1, Y2)

	# Apply depolarization noise to simulate measurement infidelities.
	meas_depol(atom1)
	meas_depol(atom2)

	#print(atom1.qstate)

	result1 = nq.measure(atom1, ns.Z)
	result2 = nq.measure(atom2, ns.Z)

	# Remap results for later convenience.
	# newresult1 should tell us how many X gates to apply in order to successfully teleport
	# our qubit using BSM; newresult2 should tell us how many Z gates to apply.

	curr_result = (result1[0], result2[0])
	if curr_result == (0, 0):
		new_result = [(0, result1[1]), (1, result2[1])]
	elif curr_result == (1, 0):
		new_result = [(0, result1[1]), (0, result2[1])]
	elif curr_result == (0, 1):
		new_result = [(1, result1[1]), (1, result2[1])]
	elif curr_result == (1, 1):
		new_result = [(1, result1[1]), (0, result2[1])]

	return new_result, [atom1, atom2]

if __name__ == '__main__':	
	nq.set_qstate_formalism(ns.QFormalism.DM)
	q1, q2 = nq.create_qubits(2)
	psiP = nq.Operator('phiP', np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], \
										 [1, 1, 1, 1]])/np.sqrt(2))
	psiM = nq.Operator('phiM', np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], \
										 [-1, -1, -1, -1]])/np.sqrt(2))
	phiP = nq.Operator('psiP', np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], \
										 [0, 0, 0, 0]])/np.sqrt(2))
	phiM = nq.Operator('psiM', np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1], \
										 [0, 0, 0, 0]])/np.sqrt(2))
	for i in [psiP, psiM, phiP, phiM]:
		nq.operate([q1, q2], i)
		print(q1.qstate)
		print(BellStateMeasurement(q1, q2))
	
