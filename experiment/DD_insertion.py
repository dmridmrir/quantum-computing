import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.models import BackendConfiguration
from qiskit.transpiler.passes import BasisTranslator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XGate, YGate, RXGate, RYGate, RZGate
from qiskit.transpiler.passes import ALAPSchedule
import sys
sys.path.append('C:/Users/wjdrm/DD_PE/src/DD')
from dynamical_decoupling import DynamicalDecoupling
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister,ClassicalRegister

# UDD 위한 세팅

def uhrig_pulse_location(k, n):
    return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2


def construct_udd_sequence(rep: int,
                           rep_gate,):
    udd_sequence = [rep_gate] * rep
    spacing = []
    for k in range(rep):
        spacing.append(uhrig_pulse_location(k, rep) - sum(spacing))
    spacing.append(1 - sum(spacing))
    return udd_sequence, spacing


#회로를 backend의 basis gate로 분리
def translate_circuit_to_basis(
    input_circuit: QuantumCircuit, configuration: BackendConfiguration
) -> QuantumCircuit:
    """Unroll the given circuit with the basis in the given configuration."""
    basis = configuration.basis_gates
    translator = BasisTranslator(SessionEquivalenceLibrary, basis)
    unrolled_dag = translator.run(circuit_to_dag(input_circuit))
    return dag_to_circuit(unrolled_dag)

#알고리즘 회로로 구현

def construct_bv_circuit(num_qubit):
    # construct the bv circuit
    qc = QuantumCircuit(num_qubit)
    qc.x(num_qubit-1)
    for i in range(num_qubit-1):
        qc.h(i)
    qc.h(num_qubit-1)
    for i in range(num_qubit-1):
        qc.cx(i, num_qubit-1)
    for i in range(num_qubit-1):
        qc.h(i)
    return qc

def custom_circuit(n_qubit):
    qc = QuantumCircuit(n_qubit)
    qc.h(0)
    for i in range(n_qubit-1):
        qc.cx(i,i+1)
        qc.x(i+1)
    return qc
    
def pea_n5():
    def cu1fixed(qc, a, b):
        qc.p(-3 * np.pi / 8, b)
        qc.cx(a, b)
        qc.p(3 * np.pi / 8, b)
        qc.cx(a, b)

    def ctu(qc, a, b):
        qc.p(-3 * np.pi / 8, b)
        qc.cx(a, b)
        qc.p(3 * np.pi / 8, b)
        qc.cx(a, b)

    qc = QuantumCircuit()

    q = QuantumRegister(5, 'q')
    c = ClassicalRegister(4, 'c')

    qc.add_register(q)
    qc.add_register(c)

    qc.h(q[0])
    qc.h(q[1])
    qc.h(q[2])
    qc.h(q[3])
    ctu(qc, q[3], q[4])
    ctu(qc, q[2], q[4])
    ctu(qc, q[2], q[4])
    ctu(qc, q[1], q[4])
    ctu(qc, q[1], q[4])
    ctu(qc, q[1], q[4])
    ctu(qc, q[1], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    ctu(qc, q[0], q[4])
    qc.h(q[0])
    qc.cp(-np.pi / 2, q[0], q[1])
    qc.h(q[1])
    qc.cp(-np.pi / 4, q[0], q[2])
    qc.cp(-np.pi / 2, q[1], q[2])
    qc.h(q[2])
    qc.cp(-np.pi / 8, q[0], q[3])
    qc.cp(-np.pi / 4, q[1], q[3])
    qc.cp(-np.pi / 2, q[2], q[3])
    qc.h(q[3])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    qc.measure(q[2], c[2])
    qc.measure(q[3], c[3])
    qc.measure(q[4], c[4])
    return qc


def multiply_n13(): 
    # 결과: (어떤 QUBIT STRING가장 큰지):
    # 몇개 결과 나오는지:
    # 어떤식으로 나오는지 (하나가 독보적으로 큰지, 여러개가 같은확률로 나오는지):
    qc = QuantumCircuit()

    q = QuantumRegister(13, 'q')
    c = ClassicalRegister(4, 'c')

    qc.add_register(q)
    qc.add_register(c)

    qc.x(q[0])
    qc.x(q[1])
    qc.x(q[2])
    qc.x(q[4])
    qc.barrier()
    qc.ccx(q[2], q[0], q[5])
    qc.ccx(q[2], q[1], q[6])
    qc.ccx(q[3], q[0], q[7])
    qc.ccx(q[3], q[1], q[8])
    qc.ccx(q[4], q[0], q[9])
    qc.ccx(q[4], q[1], q[10])
    qc.barrier()
    qc.cx(q[6], q[11])
    qc.cx(q[7], q[11])
    qc.cx(q[8], q[12])
    qc.cx(q[9], q[12])
    qc.barrier()
    qc.measure(q[5], c[0])
    qc.measure(q[11], c[1])
    qc.measure(q[12], c[2])
    qc.measure(q[10], c[3])
    return qc

def construct_hs_circuit(num_qubit):
    # construct the hs circuit
    qc = QuantumCircuit(num_qubit)
    for i in range(num_qubit):
        qc.h(i)
    for i in range(num_qubit-1):
        qc.x(i)
    for i in range(0, num_qubit-1, 2):
        qc.cz(i, i+1)
    for i in range(num_qubit-1):
        qc.x(i)
    for i in range(num_qubit):
        qc.h(i)
    for i in range(0, num_qubit-1, 2):
        qc.cz(i, i+1)
    for i in range(num_qubit):
        qc.h(i)
    return qc

def construct_graph_matrix(num_qubit,
                 coupling_map):
    # construct the graph matrix according to the hardware coupling map
    graph_matrix = np.zeros((num_qubit, num_qubit))
    for i in range(num_qubit):
        for (m, n) in coupling_map:
            if i == m and n < num_qubit:
                graph_matrix[m][n] = 1
            elif i == n and m < num_qubit:
                graph_matrix[n][m] = 1
    return graph_matrix

def convert_count_to_prob(result_counts, ideal_counts, shots):
    p_ideal = []
    p_result = []
    for key, values in result_counts.items():
        p_result.append(values / shots)
        p_ideal.append(ideal_counts.get(key, 0) / shots)
    return p_ideal, p_result

#x게이트로 통일
#하드웨어적으로 X축 회전이 더 빠르고 정확함

def theta_phi(theta, phi):
    return [RZGate(phi), RXGate(-theta), RZGate(-phi)]

def one_sequence(phi):
    sequence = []
    sequence.extend(theta_phi(np.pi, np.pi/6 + phi))
    sequence.extend(theta_phi(np.pi, phi))
    sequence.extend(theta_phi(np.pi, np.pi/2 + phi))
    sequence.extend(theta_phi(np.pi, phi))
    sequence.extend(theta_phi(np.pi, np.pi/6 + phi))
    return sequence

def kdd_sequences():
    seqences = []
    seqences.extend(one_sequence(0))
    seqences.extend(one_sequence(np.pi / 2))
    seqences.extend(one_sequence(0))
    seqences.extend(one_sequence(np.pi / 2))
    return seqences

def kdd_spacing(num_pulse=20):
    mid = 1 / num_pulse
    end = mid / 2
    spacing = []
    spacing.append(end)
    interval = [0] * 2
    for i in range(num_pulse):
        spacing.extend(interval)
        if i < num_pulse - 1:
            spacing.append(mid)
    spacing.append(end)
    return spacing



def rzx_gate_recover(circuit):

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction, qargs, cargs in circuit.data:
        if len(instruction.name) >= 3 and instruction.name[:3] == 'rzx':
            instruct_copy = instruction.copy()
            instruct_copy.name = 'rzx'
            dagcircuit.apply_operation_back(instruct_copy, qargs, cargs)
        else:
            dagcircuit.apply_operation_back(instruction.copy(), qargs, cargs)

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    circuit = dag_to_circuit(dagcircuit)
    return circuit

#패스 생성

def pm_DD_sequences(durations):
    pms = []
    DD_sequences = []

    #구현할 DD seuqnce

    hahn_X = [XGate()]
    DD_sequences.append(hahn_X)

    hahn_Y = [YGate()]
    DD_sequences.append(hahn_Y)

    CP = [XGate(), XGate()]
    DD_sequences.append(CP)

    CPMG = [YGate(), YGate()]
    DD_sequences.append(CPMG)

    xy4_sequence = [XGate(), YGate(), XGate(), YGate()]
    DD_sequences.append(xy4_sequence)

    xy8_sequence = [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()]
    DD_sequences.append(xy8_sequence)

    xy16_sequence = [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate(),
                     RXGate(-np.pi), RYGate(-np.pi), RXGate(-np.pi), RYGate(-np.pi), RYGate(-np.pi), RXGate(-np.pi),
                     RYGate(-np.pi), RXGate(-np.pi)]
    DD_sequences.append(xy16_sequence)

    #각 DD에 대해 passmanager 객체 생성

    for dd_seq in DD_sequences:
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, dd_seq)])
        pms.append(pm)

    udd_sequence1, udd_spacing1 = construct_udd_sequence(8, XGate())
    udd_sequence2, udd_spacing2 = construct_udd_sequence(8, YGate())
    kdd_spaces = kdd_spacing()
    kdd_sequence = kdd_sequences()

    sequences = [udd_sequence1, udd_sequence2, kdd_sequence]
    spaces = [udd_spacing1, udd_spacing2, kdd_spaces]

    for i in range(len(sequences)):
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, sequences[i], spacing=spaces[i])])
        pms.append(pm)

    return pms

def pm_DD_sequences_partial(durations, qubits):
    pms = []
    DD_sequences = []
    hahn_X = [XGate()]
    DD_sequences.append(hahn_X)

    hahn_Y = [YGate()]
    DD_sequences.append(hahn_Y)

    CP = [XGate(), XGate()]
    DD_sequences.append(CP)

    CPMG = [YGate(), YGate()]
    DD_sequences.append(CPMG)

    xy4_sequence = [XGate(), YGate(), XGate(), YGate()]
    DD_sequences.append(xy4_sequence)

    xy8_sequence = [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate()]
    DD_sequences.append(xy8_sequence)

    xy16_sequence = [XGate(), YGate(), XGate(), YGate(), YGate(), XGate(), YGate(), XGate(),
                     RXGate(-np.pi), RYGate(-np.pi), RXGate(-np.pi), RYGate(-np.pi), RYGate(-np.pi), RXGate(-np.pi),
                     RYGate(-np.pi), RXGate(-np.pi)]
    DD_sequences.append(xy16_sequence)

    for dd_seq in DD_sequences:
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, dd_seq, qubits=qubits,skip_reset_qubit = False)])
        pms.append(pm)

    udd_sequence1, udd_spacing1 = construct_udd_sequence(8, XGate())
    udd_sequence2, udd_spacing2 = construct_udd_sequence(8, YGate())
    kdd_spaces = kdd_spacing()
    kdd_sequence = kdd_sequences()

    sequences = [udd_sequence1, udd_sequence2, kdd_sequence]
    spaces = [udd_spacing1, udd_spacing2, kdd_spaces]

    for i in range(len(sequences)): 
        pm = PassManager([ALAPSchedule(durations),
                          DynamicalDecoupling(durations, sequences[i], spacing=spaces[i], qubits=qubits,skip_reset_qubit = False)])
        pms.append(pm)

    return pms