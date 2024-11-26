from qiskit.transpiler import PassManager, InstructionDurations
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.models import backendproperties


#시스템 경로 수정
import sys
from time import sleep
sys.path.append('')


QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="",
    set_as_default=True,
    overwrite=True
)
service = QiskitRuntimeService()
backend = service.get_backend("ibm_nazca")
configuration = backend.configuration()
properties = backend.properties()
#dt = backend.dt
#print(dt)
#print(configuration)

from DD_insertion import construct_udd_sequence, \
                                 kdd_sequences, \
                                 kdd_spacing, \
                                 construct_bv_circuit, \
                                 construct_hs_circuit, \
                                 construct_graph_matrix, \
                                 convert_count_to_prob, \
                                 translate_circuit_to_basis,\
                                 multiply_n13,\
                                 pea_n5,\
                                 shor_n5,\
                                 qtf_n4,\
                                 dnn_n8

#DD sequence적용할 알고리즘 생성

pea_circuits=[]
pea_circuits.append(pea_n5())
#qft_circuits = []
#hs_circuits = []
# qft_circuits = []
#for i in range(3, 15):
#   qft_circuits.append(construct_bv_circuit(i))

# for i in range(2, 15, 2):
#     hs_circuits.append(construct_hs_circuit(i))

# for i in range(3, 15):
#     qft_circuits.append(QFT(i))

#for circuit in qft_circuits:
#   circuit.measure_all()

# for circuit in hs_circuits:
#     circuit.measure_all()

# for circuit in qft_circuits:
#     circuit.measure_all()


# 게이트 지속시간을 x게이트로 통일
durations = InstructionDurations.from_backend(backend)
## add duration of y gates which are used for DD sequences
bconf = backend.configuration()
for i in range(bconf.num_qubits):
    x_duration = durations.get('x', i)
    durations.update(InstructionDurations(
        [('y', i, x_duration)]
        ))

    durations.update(InstructionDurations(
        [('rx', i, x_duration)]
        ))

    durations.update(InstructionDurations(
        [('ry', i, x_duration)]
        ))
    durations.update(InstructionDurations(
        [('if_else',None,0)]
        ))

#graph_state_circuits = []
#coupling_map = backend.configuration().coupling_map

#for i in range(3, 15):
#    gs_circuit_matrix = construct_graph_matrix(i, coupling_map)
#    graph_state_circuits.append(GraphState(gs_circuit_matrix))

#for circuit in graph_state_circuits:
#    circuit.measure_all()

def wait_for_pending_jobs_to_complete(service, backend):
    print("백엔드에서 대기 중인 작업이 완료될 때까지 대기")
    while True:
        jobs = service.jobs()  # 모든 작업을 가져옵니다.
        queued_jobs = [job for job in jobs if job.backend() == backend and job.status().name == 'QUEUED']
        
        if len(queued_jobs) < 3:
            break
            
        print("잠시 대기")
        sleep(10)

timing_constraints = backend.configuration().timing_constraints
pulse_alignment = timing_constraints['pulse_alignment']

for i in range(10):      # 평균을 내기 위해선 10으로 설정해두면 됌.

    ##DD sequence 적용
    from DD_insertion import pm_DD_sequences
    pms = pm_DD_sequences(durations,pulse_alignment)

    pea_job_ids = []
    pea_jobs = []

    for circuit in pea_circuits:
        circuit_list = []
        #트랜스파일 전 회로 : circuit``
        #트랜스파일 후 회로 : transpiled circuit
        #두개 비교해서 어떻게 게이트가 변환되는지 확인
        transpiled_qc = transpile(circuit, backend=backend, optimization_level=0, seed_transpiler=1)
        circuit_list.append(transpiled_qc)
        qc_transpile = pms[-1].run(transpiled_qc)

        for pm in pms:
            #transpiled circuit과 qc_trans pile_base비교해서 sequence삽입 확인
            qc_transpile = pm.run(transpiled_qc)
            qc_transpile_base = translate_circuit_to_basis(qc_transpile, bconf)
            print(qc_transpile_base)
            
            circuit_list.append(qc_transpile_base)
        wait_for_pending_jobs_to_complete(service, backend)
        job = backend.run(circuit_list, shots=8192)
        pea_jobs.append(job)
        pea_job_ids.append(job.job_id())

    print(pea_job_ids)