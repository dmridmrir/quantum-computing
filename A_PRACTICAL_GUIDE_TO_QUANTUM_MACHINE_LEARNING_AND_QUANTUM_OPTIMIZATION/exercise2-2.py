#Construct the circuit in Figure 2.3b. Draw the result and use the output to verify
#whether your circuit implementation is correct.


from qiskit import *
import matplotlib.pyplot as plt
import numpy as np

q1 = QuantumRegister(1,"q1")
q2 = QuantumRegister(1,"q2")
qc = QuantumCircuit(q1,q2)

qc.z(q1)
qc.y(q2)
qc.cry(np.pi/2,q1,q2)
qc.u(np.pi/4,np.pi,0,0)
qc.rz(np.pi/4,1)

qc.draw('mpl',style="iqp",interactive=True)
plt.show()
