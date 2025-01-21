
from setup_and_utils import *
from training_data import *


#QCBM Circuit
from scipy.special import comb

total_qubits = n_qubits + n_ancillas
dev = qml.device("default.qubit",wires=total_qubits)

 
#QCBM Circuit - RZ + IsingXY + IsingZZ    
def qcbm_circuit(params,total_qubits):
    
    rz_params = params[:total_qubits]
    ising_params = params[total_qubits:]
    for i in range(total_qubits):
        qml.RZ(rz_params[i],wires=i)
    for i in range(total_qubits-1):
        qml.IsingXY(ising_params[i],wires=[i,i+1])
    qml.IsingXY(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    for i in range(total_qubits-1):
        qml.IsingZZ(ising_params[i],wires=[i,i+1])
    qml.IsingZZ(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    
folds = 8
# # Initialize a JAX random key
# key = jax.random.PRNGKey(0)
# # Generate initial parameters as a JAX array
# initial_params = jax.random.uniform(key, shape=(folds, 3 * total_qubits), minval=0.0, maxval=1.0)

np.random.seed(0)  # Set seed for reproducibility
initial_params = np.random.uniform(low=0.0, high=1.0, size=(folds, 3 * total_qubits))


#QCBM Circuit - RX + RZ + CNOT    
# def qcbm_circuit(params,total_qubits):
    
#     rz_params = params[:total_qubits]
#     ising_params = params[total_qubits:]
#     for i in range(total_qubits):
#         qml.RX(ising_params[i],wires=i)
#         qml.RZ(rz_params[i],wires=i)
#     for i in range(total_qubits-1):
#         qml.CNOT(wires=[i,i+1])
#     qml.CNOT(wires=[total_qubits-1,0])
    
# folds = 8
# # Initialize a JAX random key
# key = jax.random.PRNGKey(0)
# # Generate initial parameters as a JAX array
# initial_params = jax.random.uniform(key, shape=(folds, 2 * total_qubits), minval=0.0, maxval=1.0)


odd_wires = list(i for i in range(total_qubits) if i%2 != 0)
# @qml.qnode(dev,interface='jax')
@qml.qnode(dev)
def circuit(input_params,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):

    #Random state for pretraining
    # uniform_init(num_qubits,distribution)
    qml.BasisState(jnp.zeros(total_qubits, dtype=jnp.int32), wires=list(range(total_qubits)))
    
    for i in range(total_qubits):
        if i%2 == 0:
            qml.X(i)
        
    for i in range(8):
        # qml.Barrier(range(total_qubits))
        qcbm_circuit(params=input_params[i],total_qubits=total_qubits)
    
    ### Measurement to find anti-cat
    # output = qml.probs(wires=list(i for i in range(total_qubits) if i%2 != 0))
    # return output
    
    ###Measurement for decoder without the shuffle
    # output1 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 != 0))
    # output2 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 == 0))
    # return [output1,output2]
    
    ###Measurement for decoder with the shuffle
    # output1 = qml.probs(wires=list(i for i in range(n_qubits)))
    # output2 = qml.probs(wires=list(i for i in range(n_qubits,total_qubits)))
    # return [output1,output2]

    ###Measurement for fidelity_model
    return qml.density_matrix(wires=odd_wires)
