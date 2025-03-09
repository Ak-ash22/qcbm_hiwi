
from setup_and_utils import *
from training_data import *


#QCBM Circuit
from scipy.special import comb

total_qubits = n_qubits + n_ancillas + n_extra_qubits
dev = qml.device("default.qubit",wires=total_qubits)

 
#QCBM Circuit - RZ + IsingXY + IsingZZ    
def qcbm_circuit(params,total_qubits):
    
    rz_params = params[:total_qubits]
    ising_params1 = params[total_qubits:2*total_qubits]
    ising_params2 = params[2*total_qubits:]
    
    for i in range(total_qubits):
        qml.RZ(rz_params[i],wires=i)
    for i in range(total_qubits-1):
        qml.IsingXY(ising_params1[i],wires=[i,i+1])
    qml.IsingXY(ising_params1[-1],wires=[total_qubits-1,0])
    for i in range(total_qubits-1):
        qml.IsingZZ(ising_params2[i],wires=[i,i+1])
    qml.IsingZZ(ising_params2[-1],wires=[total_qubits-1,0])
    
    
folds = 6
# # Initialize a JAX random key
key = jax.random.PRNGKey(0)
# # Generate initial parameters as a JAX array
# initial_params = jax.random.uniform(key, shape=(folds, (4 * total_qubits) - 1), minval=0.0, maxval=1.0)
initial_params = jax.random.uniform(key, shape=(folds, (3 * total_qubits)), minval=0.0, maxval=1.0)

@qml.qnode(dev,interface='jax')
# def circuit(input_params,folds,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):
def circuit(input_params,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):

    #Random state for pretraining
    # uniform_init(num_qubits,distribution)
    qml.BasisState(jnp.zeros(total_qubits, dtype=jnp.int32), wires=list(range(total_qubits)))
    
    for i in range(n_qubits+n_ancillas): 
        if i%2 == 0:
            qml.X(i)

    for i in range(folds):
        # qml.Barrier(range(total_qubits))
        qcbm_circuit(params=input_params[i],total_qubits=total_qubits)
    
    ### Measurement to find cat
    # output = qml.probs(wires=list(i for i in range(total_qubits) if i%2 == 0))
    # return output
    
    ### Measurement to find anti-cat
    # output = qml.probs(wires=list(i for i in range(total_qubits) if i%2 == 0))
    # return output
    
    ###Measurement for decoder1 without the shuffle
    output1 = qml.probs(wires=list(i for i in range(n_qubits+n_ancillas) if i%2 == 0)) #cat
    output2 = qml.probs(wires=list(i for i in range(n_qubits+n_ancillas) if i%2 != 0)) #anticat
    return [output1,output2]
    