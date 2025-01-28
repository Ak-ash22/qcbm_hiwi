
from setup_and_utils import *
from training_data import *


#QCBM Circuit
from scipy.special import comb

total_qubits = n_qubits + n_ancillas
dev = qml.device("default.qubit",wires=total_qubits)

#Load the trained cat model

with open('/home/akashm/PROJECT/Hiwi_qml/qcbm/cat_anticat/4model2_cat_anticat_distribution.pkl',"rb") as file:
    params = pickle.load(file)

final_epoch = params['final_epoch'],
kl_div = params['divs']
trained_params = params['parameters']

optimal_params = trained_params[np.argmin(kl_div)]

 
#QCBM Circuit - RZ + IsingXY + IsingZZ    
def qcbm_circuit(params,total_qubits=total_qubits):
    
    rz_params = params[3*n_qubits:3*n_qubits+total_qubits]
    ising_params = params[3*n_qubits+total_qubits:]
    for i in range(total_qubits):
        qml.RZ(rz_params[i],wires=i)
    for i in range(total_qubits-1):
        qml.IsingXY(ising_params[i],wires=[i,i+1])
    qml.IsingXY(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    
    ##For Decoder2 for faster convergence
    for i in range(total_qubits-2):
        qml.IsingXY(ising_params[total_qubits+i],wires=[i,i+2])
        
    for i in range(total_qubits-1):
        qml.IsingZZ(ising_params[i],wires=[i,i+1])
    qml.IsingZZ(ising_params[total_qubits-1],wires=[total_qubits-1,0])
    
    
    
folds = 8
# Initialize a JAX random key
key = jax.random.PRNGKey(0)
# Generate initial parameters as a JAX array
# initial_params = jax.random.uniform(key, shape=(folds, (4 * total_qubits) - 1), minval=0.0, maxval=1.0)
initial_params = jax.random.uniform(key, shape=(folds, (3 * n_qubits)+(3*total_qubits)), minval=0.0, maxval=1.0)


# QCBM Circuit - RX + RZ + CNOT    
def anticat_circuit(params,total_qubits=n_qubits):
    
    rz_params = params[:total_qubits]
    ising_params = params[total_qubits:]
    for i in range(total_qubits):
        qml.RX(rz_params[i],wires=n_qubits+i)
    for i in range(n_qubits,n_qubits+total_qubits-1):
        qml.CNOT(wires=[i,i+1])
    qml.CNOT(wires=[n_qubits+total_qubits-1,n_qubits])
    # for i in range(total_qubits):
    #     qml.RZ(rz_params[i],wires=n_qubits+i)
    # # for i in range(total_qubits):
    # #     qml.RZ(rz_params[i],wires=n_qubits+i)
    # for i in range(n_qubits,n_qubits+total_qubits-1):
    #     qml.IsingXY(ising_params[i],wires=[i,i+1])
    # qml.IsingXY(ising_params[i],wires=[n_qubits+total_qubits-1,n_qubits])
    # for i in range(n_qubits,n_qubits+total_qubits-1):
    #     qml.IsingZZ(ising_params[i],wires=[i,i+1])
    # qml.IsingZZ(ising_params[i],wires=[n_qubits+total_qubits-1,n_qubits])

#Loading the original cat training data
with open('/home/akashm/PROJECT/Hiwi_qml/qcbm/model_local_no_pretraining/4qubit_target_distribution.pkl',"rb") as file1:
    cat_data = pickle.load(file1)
# new_cat_data = cat_data[::-1]
cat_data = cat_data/np.linalg.norm(cat_data)

# odd_wires = list(i for i in range(total_qubits) if i%2 != 0)
@qml.qnode(dev,interface='jax')
# @qml.qnode(dev)
def circuit(input_params,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):

    #Random state for pretraining
    # uniform_init(num_qubits,distribution)
    # qml.BasisState(jnp.zeros(total_qubits, dtype=jnp.int32), wires=list(range(total_qubits)))
    
    #Initialize new cat
    qml.QubitStateVector(cat_data,wires=list(i for i in range(n_qubits)))
    qml.BasisState(jnp.zeros(n_qubits, dtype=jnp.int32), wires=list(range(n_qubits,total_qubits)))

    
    #Circuit for Anticat intialization
    for i in range(3):
        anticat_circuit(input_params[i])

    # for i in range(total_qubits):
    #     if i%2 == 0:
    #         qml.X(i)
    
    ###Random initial state for decoder2
    # for i in range(total_qubits):
    #     if np.random.randint(0, 2) == 1:
    #         qml.X(i)
        
    for i in range(8):
        # qml.Barrier(range(total_qubits))
        qml.adjoint(qcbm_circuit)(params=input_params[i,:])
    
    ## Measurement to find anti-cat
    output = qml.probs(wires=range(total_qubits))
    return output
    
    ###Measurement for decoder1 without the shuffle
    # output1 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 != 0))
    # output2 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 == 0))
    # return [output1,output2]
    
    ##Measurement for decoder2 with the shuffle
    # output1 = qml.probs(wires=list(i for i in range(n_qubits)))
    # output2 = qml.probs(wires=list(i for i in range(n_qubits,total_qubits)))
    # return [output1,output2]

    ###Measurement for fidelity_model
    # return qml.density_matrix(wires=odd_wires)
