from setup_and_utils import *

n_qubits = 10
n_ancillas = 8

def create_target_distribution(n_qubits):
    x_full = jnp.arange(0,2**n_qubits,dtype=jnp.float64)
    x_split = jnp.array_split(x_full,4)

    y1 = relu(x_split[0])
    y2 = sigmoid(x_split[1] - x_split[1][0])
    y3 = elu(x_split[2] - x_split[2][0])
    y4 = tanh(x_split[3] - x_split[3][0])
    
    # target_distribution = gaussian(x_full)

    target_distribution = jnp.concatenate([y1,y2,y3,y4], dtype=jnp.float64)
    target_distribution /= trapezoid(target_distribution)
    return target_distribution

#Particle number distribution
def pnumber_distribution(distribution, n_qubits):
    p_distribution = jnp.zeros(n_qubits+1,dtype=jnp.float64)
    for i in range(2**n_qubits):
        binary_string = format(i,f'0{n_qubits}b')
        num_of_ones = binary_string.count('1')
        p_distribution = p_distribution.at[num_of_ones].add(distribution[i])
    #Normalize the distribution
    p_distribution/=p_distribution.sum()
    return p_distribution

def save_distribution(filename, target_distribution):
    with open(filename, "wb") as f:
        pickle.dump(target_distribution, f)

target_distribution = create_target_distribution(n_qubits)
targetp_distribution = pnumber_distribution(target_distribution, n_qubits)

if __name__ == "__main__":
    save_distribution("model_copy/target_distribution.pkl", target_distribution)
    save_distribution("model_copy/pnumber_distribution.pkl", targetp_distribution)
    print("Target distribution and Particle Number distribution saved!")