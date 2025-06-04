import torch
import numpy as np




class QuantumGates:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def Z():
        return torch.tensor([[1,0],[0,-1]],dtype=torch.complex64,device=QuantumGates.device)
    
    @staticmethod
    def X():
        return torch.tensor([[0,1],[1,0]],dtype=torch.complex64,device=QuantumGates.device)
    
    @staticmethod
    def Y():
        return 1j*torch.tensor([[0,-1],[1,0]],dtype=torch.complex64,device=QuantumGates.device)
    
    @staticmethod
    def Rx(theta):
        return torch.cos(theta/2)*torch.eye(2,dtype=torch.complex64,
                                            device=QuantumGates.device) -1j*torch.sin(theta/2)*QuantumGates.X()

    @staticmethod
    def Ry(theta):
        return torch.cos(theta/2) * torch.eye(2,dtype=torch.complex64,
                                              device=QuantumGates.device) -1j* torch.sin(theta/2) *QuantumGates.Y()

    @staticmethod
    def Rz(theta):
        return torch.cos(theta/2) * torch.eye(2,dtype=torch.complex64,device=
                                              QuantumGates.device) - 1j*torch.sin(theta/2) * QuantumGates.Z()
    
    @staticmethod
    def NeighborCNOT():
        I = torch.eye(2,dtype=torch.complex64,device=QuantumGates.device)
        zero_proj = torch.tensor([[1,0],[0,0]],dtype=torch.complex64,device=QuantumGates.device)
        one_proj = torch.tensor([[0,0],[0,1]],dtype=torch.complex64,device=QuantumGates.device)
        return torch.kron(zero_proj,I) + torch.kron(one_proj,QuantumGates.X())
    
    @staticmethod
    def CNOT(control,target,n_qubits):
        Pro0 = [torch.eye(2,dtype=torch.complex64,device=QuantumGates.device)
               for i in range(n_qubits)]
        Pro0[control] = torch.tensor([[1,0],[0,0]],dtype=torch.complex64,
                                    device=QuantumGates.device)
        Pro1 = [torch.eye(2,dtype=torch.complex64,device=QuantumGates.device)
               for i in range(n_qubits)]
        Pro1[control] = torch.tensor([[0,0],[0,1]],dtype=torch.complex64,
                                    device=QuantumGates.device)
        Pro1[target] = QuantumGates.X()
        return utils.n_kron(Pro0) + utils.n_kron(Pro1)


class utils:

    def n_kron(ops_to_kron):
        res = ops_to_kron[0]
        for element in ops_to_kron[1:]:
            res = torch.kron(res,element)
        return res
    

    def apply_one_site(site,state,op):
        state = state.reshape(2**(site),2,-1)
        state = torch.tensordot(op,state,dims=([1],[1]))
        state = state.permute(1,0,2).contiguous()
        return state.reshape(-1,)
    
    def apply_two_site(sites,state,op):
        state = state.reshape(2**(sites[0]),2,2**(sites[1]-sites[0]-1),2,-1)
        op = op.view(2,2,2,2)
        state = torch.tensordot(op,state,dims=([2,3],[1,3]))
        state = state.permute(2,0,3,1,4).contiguous()
        return state.reshape(-1,)

    def measure_expectation(state,op,site):
        state = state.view(2**(site),2,-1)
        measured_state = torch.tensordot(op,state,dims=([1],[1]))
        expectation_value = torch.vdot(state.view(-1,),measured_state.view(-1,)).real 
        return expectation_value
        
    def get_probs(state):
        probs = torch.abs(state)**2
        return probs.real
    

class Circuits:

    def StronglyEntanglingLayer(params,state,n_qubits=8):
        for i in range(n_qubits):
            Ry_gate = QuantumGates.Ry(theta=params[i])
            state = utils.apply_one_site(i,state,op=Ry_gate)
        for i in range(n_qubits-1):
            cnot = QuantumGates.CNOT(control=i,target=i+1,n_qubits=n_qubits)
            state = cnot@state
        cnot = QuantumGates.CNOT(control=n_qubits-1,target=0,n_qubits=n_qubits)
        state = cnot@state    
        return state
    

    

    