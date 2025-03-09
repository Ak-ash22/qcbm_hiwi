from training_data import *
from scipy.special import comb

total_qubits = n_qubits + n_ancillas
dev = qml.device("default.qubit",wires=total_qubits)

#QCBM Circuit - RZ + IsingXY + IsingZZ    
def qcbm_circuit(params,total_qubits=total_qubits):
    
    rz_params = params[:total_qubits]
    ising_params1 = params[total_qubits:2*total_qubits-1]
    ising_params2 = params[2*total_qubits-1:]
    
    
    for i in range(total_qubits):
        qml.RZ(rz_params[i],wires=i)
    for i in range(total_qubits-1):
        qml.IsingXY(ising_params1[i],wires=[i,i+1])
    qml.IsingXY(ising_params1[-1],wires=[total_qubits-1,0])
    for i in range(total_qubits-1):
        qml.IsingZZ(ising_params2[i],wires=[i,i+1])
    qml.IsingZZ(ising_params2[-1],wires=[total_qubits-1,0])
    
    
    
# # Define folds for each part of the circuit
# vqc_folds = 3
# qcbm_folds = 6

# # Initialize a JAX random key
# key = jax.random.PRNGKey(0)
# key_vqc, key_qcbm = jax.random.split(key)  # Split the random key for different parts

# # Generate random parameters for VQC circuit
# vqc_params = jax.random.uniform(key_vqc, shape=(vqc_folds, 3 * n_qubits), minval=0.0, maxval=1.0)
# # Generate random parameters for QCBM circuit
# qcbm_params = jax.random.uniform(key_qcbm, shape=(qcbm_folds, 3 * total_qubits), minval=0.0, maxval=1.0)

# # Concatenate along the first axis (stacking the parameters)
# initial_params = (vqc_params, qcbm_params)


def vqc_circuit(params,total_qubits=n_qubits):
    
    rz_params = params[:n_qubits]
    ising_params1 = params[n_qubits:2*n_qubits]
    ising_params2 = params[2*n_qubits:]

    for i in range(total_qubits):
        qml.RZ(rz_params[i],wires=i)
    for i in range(n_qubits-1):
        qml.IsingXY(ising_params1[i],wires=[i,i+1])
    qml.IsingXY(ising_params1[-1],wires=[n_qubits-1,0])
    for i in range(n_qubits-1):
        qml.IsingZZ(ising_params2[i],wires=[i,i+1])
    qml.IsingZZ(ising_params2[-1],wires=[n_qubits-1,0])
    
    ##Circuit - RX + RZ + CNOT    
    # rx_params = params[:n_qubits]
    # ry_params = params[n_qubits:2*n_qubits]
    # rz_params = params[2*n_qubits:]
    
    # for i in range(total_qubits):
    #     qml.RX(rx_params[i],wires=i)
    #     qml.RY(ry_params[i],wires=i)
    #     qml.RZ(rz_params[i],wires=i)
    # for i in range(total_qubits-1):
    #     qml.CNOT(wires=[i,i+1])
    # qml.CNOT(wires=[total_qubits-1,0])
    


# #Loading the original cat training data
with open('/home/akashm/PROJECT/qcbm_hiwi/qcbm_project/cat_anticat/data/three_particle_distribution.pkl',"rb") as file1:
    cat_data = pickle.load(file1)
cat_data = cat_data/np.linalg.norm(cat_data)

with open('/home/akashm/PROJECT/qcbm_hiwi/qcbm_project/cat_anticat/data/three_particle_avg_anticat_distribution.pkl',"rb") as file2:
    anticat_data = pickle.load(file2)
anticat_data = anticat_data/np.linalg.norm(anticat_data)
# anticat_data = jnp.array(anticat_data)  # Convert to non-traced JAX array


#Particle Number based pretraining
def uniform_init(num_qubits,distribution,seed):
    
    all_states = np.array([format(i, f'0{num_qubits}b') for i in range(2**num_qubits)])
    amps = jnp.zeros(len(all_states),dtype=jnp.float64)
    
    #uniform distribution of all possible states
    for i in range(len(all_states)):
        k = all_states[i]
        num_of_ones = k.count('1')
        number_of_states_with_num_of_ones = comb(num_qubits, num_of_ones)
        amps = amps.at[i].set(distribution[num_of_ones] / number_of_states_with_num_of_ones)      
    amps = jnp.sqrt(amps)
    
    if seed == 0:        
        qml.StatePrep(amps,wires=range(num_qubits))
    else:
        qml.StatePrep(amps,wires=range(n_qubits,n_qubits+num_qubits))

cat_pre_training = pnumber_distribution(cat_data, n_qubits)
anticat_pre_training = pnumber_distribution(anticat_data, n_qubits)


@qml.qnode(dev,interface='jax')
def circuit(input_params,vqc_folds,qcbm_folds,num_qubits=n_qubits,ancilla_qubits=n_ancillas,total_qubits=total_qubits):
       
    vqc_params = input_params[0]
    qcbm_params = input_params[1]
    
    #Top Half of the circuit -- Cat Pretraining +VQC
    uniform_init(num_qubits,cat_pre_training,seed=0)
    # qml.BasisState(jnp.zeros(n_qubits, dtype=jnp.int32), wires=list(range(n_qubits)))

    for i in range(vqc_folds):
        vqc_circuit(vqc_params[i])
        
    #Bottom Half of the circuit -- Anticat
    qml.QubitStateVector(jax.lax.stop_gradient(anticat_data),wires=list(i for i in range(n_qubits,total_qubits)))
    # uniform_init(num_qubits,anticat_pre_training,seed=1)
    
    #Adjoint of QCBM circuit
    for i in range(qcbm_folds):
        qml.adjoint(qcbm_circuit)(params=qcbm_params[i,:])
        
    ##Measurement of all qubits
    output1 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 != 0))
    output2 = qml.probs(wires=list(i for i in range(total_qubits) if i%2 == 0))
    return [output1, output2]

