import torch
import numpy as np
import math

# def view_as_qubits(state: torch.Tensor, target_axes: int) -> torch.Tensor:
#     """Return a view with at most 15+1 dims (target + packed rest)."""
#     import math
#     n = int(math.log2(state.numel()))
#     if n <= 15:
#         return state.view(*([2] * n))
#     # pack “spectator” qubits into one axis
#     packed = 1 << (n - 15)            # 2**(n-15)
#     return state.view(packed, *([2] * 15))




class QuantumGates:

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

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
    
    # @staticmethod
    # def CNOT(control,target,n_qubits):
    #     Pro0 = [torch.eye(2,dtype=torch.complex32,device=QuantumGates.device)
    #            for i in range(n_qubits)]
    #     Pro0[control] = torch.tensor([[1,0],[0,0]],dtype=torch.complex64,
    #                                 device=QuantumGates.device)
    #     Pro1 = [torch.eye(2,dtype=torch.complex64,device=QuantumGates.device)
    #            for i in range(n_qubits)]
    #     Pro1[control] = torch.tensor([[0,0],[0,1]],dtype=torch.complex64,
    #                                 device=QuantumGates.device)
    #     Pro1[target] = QuantumGates.X()
    #     return utils.n_kron(Pro0) + utils.n_kron(Pro1)
    @staticmethod
    def CNOT(state,control,target):
        n = int(math.log2(state.numel()))

        if n<=14:
            psi = state.reshape(*([2]*n))
            psi = psi.movedim([control,target],[0,1])
            low = psi[0]
            high = psi[1]
            high = high.flip(0)
            psi = torch.stack((low,high),dim=0)\
                .movedim([0,1],[control,target])
            return psi.reshape(-1).contiguous()
        
        lo, hi = sorted((control, target))
        before = 1 << lo
        mid    = 1 << (hi - lo - 1)
        after  = 1 << (n - hi - 1)

        # shape: (before, 2[lo], mid, 2[hi], after)  ≤ 5 axes
        psi = state.view(before, 2, mid, 2, after)

        if control < target:
            ctrl_axis, targ_axis = 1, 3
        else:
            ctrl_axis, targ_axis = 3, 1

        # pull the two 2-axes to front -> (control, target, rest)
        psi = psi.movedim([ctrl_axis, targ_axis], [0, 1])
        low  = psi[0]            # control = 0
        high = psi[1].flip(0)    # control = 1 → flip target
        psi  = torch.stack((low, high), 0).movedim([0, 1], [ctrl_axis, targ_axis])
        return psi.reshape(-1).contiguous()






class utils:

    def n_kron(ops_to_kron):
        res = ops_to_kron[0]
        for element in ops_to_kron[1:]:
            res = torch.kron(res,element)
        return res
    

    def apply_one_site(site,state,op):
        # state = state.reshape(2**(site),2,-1)
        # state = torch.tensordot(op,state,dims=([1],[1]))
        # state = state.permute(1,0,2).contiguous()
        # return state.reshape(-1,)
        n = int(math.log2(state.numel()))

        if n<=15:
            psi = state.view(*([2] * n)) 
            psi = psi.movedim(site, 0)
            psi = op @ psi.reshape(2,-1)
            psi = psi.reshape(2, *([2] * (n - 1))).movedim(0, site)
            return psi.reshape(-1).contiguous()
        
        before = 1 << site                  # 2**site
        after  = 1 << (n - site - 1)        # 2**(n-site-1)

        psi = state.view(before, 2, after)             # (before, 2, after)
        # Bring the '2' axis to the front so we can matmul in one GEMM
        psi = psi.permute(1, 0, 2).reshape(2, -1)      # (2, before*after)
        psi = op @ psi                                # apply gate
        psi = psi.reshape(2, before, after).permute(1, 0, 2)
        return psi.reshape(-1).contiguous()
    
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
    

    

    