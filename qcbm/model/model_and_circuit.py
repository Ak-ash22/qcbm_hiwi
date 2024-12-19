
from setup_and_utils import *
from training_data import *


#QCBM Circuit
from scipy.special import comb

total_qubits = n_qubits + n_ancillas
dev = qml.device("default.qubit",wires=total_qubits)

#Uniform initialization of the quantum state
def uniform_init(num_qubits,distribution=targetp_distribution):
    
    all_states = np.array([format(i, f'0{num_qubits}b') for i in range(2**num_qubits)])
    amps = jnp.zeros(len(all_states),dtype=jnp.float64)
    #uniform distribution of all possible states
    for i in range(len(all_states)):
        k = all_states[i]
        num_of_ones = k.count('1')
        number_of_states_with_num_of_ones = comb(num_qubits, num_of_ones)
        amps = amps.at[i].set(distribution[num_of_ones] / number_of_states_with_num_of_ones)      
    amps = jnp.sqrt(amps)        
    qml.StatePrep(amps,wires=range(num_qubits))
    
    
def qcbm_circuit(params,total_qubits):
    
    rz_params = params[:total_qubits]
    ising_params = params[total_qubits:]
    for i in range(total_qubits):
        # qml.RX(ising_params[i],wires=i)
        qml.RZ(rz_params[i],wires=i)

    # for i in range(total_qubits-1):
    #     qml.CNOT(wires=[i,i+1])
    # qml.CNOT(wires=[total_qubits-1,0])
    
    for i in range(total_qubits-1):
        qml.IsingXY(ising_params[i],wires=[i,i+1])
    qml.IsingXY(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    
    for i in range(total_qubits-1):
        qml.IsingZZ(ising_params[i],wires=[i,i+1])
    qml.IsingZZ(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    

folds = 10

# Initialize a JAX random key
key = jax.random.PRNGKey(0)

# Generate initial parameters as a JAX array
initial_params = jax.random.uniform(key, shape=(folds, 3 * total_qubits), minval=0.0, maxval=1.0)

    

@qml.qnode(dev, interface='jax')
def circuit(input_params,distribution=targetp_distribution,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):

    #Random state for pretraining
    uniform_init(num_qubits,distribution)
    
    for i in range(10):
        qml.Barrier(range(total_qubits))
        qcbm_circuit(params=input_params[i],total_qubits=total_qubits)

    return qml.probs(wires=[0,1,2,3,4,5,6,7,8,9])
