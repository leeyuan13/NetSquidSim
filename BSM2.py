import netsquid as ns
import netsquid.qubits as nq
import numpy as np

# NetSquid has ns.CNOT in-built.

def BellStateMeasurement(atom1, atom2):
	# First do a CNOT on 1 and 2.
	nq.operate([atom1, atom2], ns.CNOT)
	
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
	
