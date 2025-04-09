# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:36:54 2022

Program to perform Pauli sampling to compute von Neumann Stabilizer entropy and magic capacity for arbitrary states
Tested up to 24 qubits

Companion code for "Efficient mutual magic and magic capacity with matrix product states" 
by Poetri Sonya Tarabunga and Tobias Haug

Works by using Pauli sampling to sample from Pauli spectrum P(\sigma)=2^-N <psi|\sigma|psi>^2
Pauli sampling is done by sampling in computational basis from U_bell^{\otimes N} |psi^*>\otimes|psi>
which is equivalent to sampling from P(\sigma), where U_bell is the Bell transformation.
Use this to estimate von Neumann SE -\sum_\sigma P(\sigma) log(<psi|\sigma|psi>^2)
Can also sample as a Bell measurement from U_bell^{\otimes N} |psi>\otimes|psi>

Main problem is that |psi^*>\otimes|psi> is too large to be stored in memory
Thus, first we use Feynman like approach where we store only |psi> and describe sampling trajectory
of tensor product as a superposition (see manuscript for details)

This program uses this Feynmann-like trajectory approach to sample the first half of the qubits, 
then switches to direct sampling from marginals once enough qubits have been sampled.

This way, scaling of O((2\sqrt{2})^N) is achieved and at least 24 qubits,
compared to previous best method of O(8^N) which was limited to ~15 qubits.


@author: Tobias Haug, 
Technology Innovation Institute, UAE
tobias.haug@u.nus.edu


"""



n_qubits=6 ##number of qubits

##type of state to sample,
# 0: 0 state, 
#1: N-qubit T-state 
#2: haar random
# 3: GUE evolution
# 4: random Clifford +T gates
model=1

n_sample=10**3 ##number of samples to take

evolution_time=1 ##evolution_time for random Hamiltonian evolution for model=3
n_tgates=1 ##number of T-gates for model=4
layers=100 ##number of layers for model=4

#max_qubit controls at what qubit number to switch from Feynmann to direct sampling
max_qubit=min((n_qubits-4)//2,12) 


do_bell_sampling=False ##whether to sample from Pauli distribution or Bell distribution


import qutip as qt
import numpy as np
import time
import operator
from functools import reduce




#tensors operators together 
def genFockOp(op,position,size,levels=2,opdim=0):
    opList=[qt.qeye(levels) for x in range(size-opdim)]
    opList[position]=op
    return qt.tensor(opList)


def numberToBase(n, b,n_qubits):
    if n == 0:
        return np.zeros(n_qubits,dtype=int)
    digits = np.zeros(n_qubits,dtype=int)
    counter=0
    while n:
        digits[counter]=int(n % b)
        n //= b
        counter+=1
    return digits[::-1]


    
def get_all_paulis_N(n_qubits):
    ##get all pauli operators
    n_paulis=4**n_qubits

    pauli_list=np.zeros([n_paulis,n_qubits],dtype=int)
    for k in range(n_paulis):
        pauli_list[k,:]=numberToBase(k,4,n_qubits)

    levels=2
    opZ=[genFockOp(qt.sigmaz(),i,n_qubits,levels) for i in range(n_qubits)]
    opX=[genFockOp(qt.sigmax(),i,n_qubits,levels) for i in range(n_qubits)]
    opY=[genFockOp(qt.sigmay(),i,n_qubits,levels) for i in range(n_qubits)]

    
    opId=genFockOp(qt.qeye(levels),0,n_qubits)

    pauli_op_list=[]
    for k in range(n_paulis):
        pauli_string=pauli_list[k]
        pauli=opId
        for i in range(n_qubits):
            if(pauli_string[i]==1):
                pauli=opX[i]*pauli
            elif(pauli_string[i]==2):
                pauli=opY[i]*pauli
            elif(pauli_string[i]==3):
                pauli=opZ[i]*pauli
                
        pauli_op_list.append(pauli)
    return pauli_op_list,pauli_list





def get_conversion_matrix_mod_add_index(base_states):
    ##precompuation for computing renyi stabilizer entropy
    n_qubits=len(base_states[0])
    mag = len(base_states)
    to_index=2**np.arange(n_qubits)[::-1]
    conversion_matrix=np.zeros([mag,mag],dtype=int)
    for j_count in range(mag):
        base_j=base_states[j_count]
        k_plus_j=np.mod(base_states+base_j,2)
        k_plus_j_index=np.sum(k_plus_j*to_index,axis=1)
        conversion_matrix[j_count,:]=k_plus_j_index

    return conversion_matrix

def get_conversion_matrix_binary_prod(base_states):
    ##precompuation for computing renyi stabilizer entropy
    mag = len(base_states)
    conversion_matrix=np.zeros([mag,mag],dtype=int)
    
        
    for i_count in range(mag):
        base_i=base_states[i_count]
        binary_product=np.mod(np.dot(base_states,base_i),2)
        conversion_matrix[i_count,:]=(-1)**binary_product
        

    return conversion_matrix

def renyi_entropy_fast(state,conversion_matrix_mod_add_index,conversion_matrix_binary_prod,alpha=[2]):
    ##fast code to compute renyi entropy which uses numpy to full extent
    #requires precomputation with get_conversion_matrix_binary_prod and get_conversion_matrix_mod_add_index
    coeffs=state.data.toarray()[:,0]
    n_qubits=len(state.dims[0])

       
    subtract_log=n_qubits*np.log(2)
    
    prob_list_full=np.abs(np.dot(np.conjugate(coeffs)*conversion_matrix_binary_prod, coeffs[conversion_matrix_mod_add_index] ))**2
    
    #print(prob_list_full)
    epsilon_cutoff=10**-30
    prob_list=2**(-n_qubits)*prob_list_full[prob_list_full>epsilon_cutoff]
    

    renyi_fast_list=[]
    for alpha_p in alpha:
        if(alpha_p==1): ##shannon magic
            renyi_fast= -np.sum(prob_list*np.log(prob_list))-subtract_log
        else:
        

            renyi_fast = 1/(1-alpha_p)*np.log(np.sum(prob_list**(alpha_p)))-subtract_log

        renyi_fast_list.append(renyi_fast)
        
    magic_capacity=np.sum(prob_list*(-np.log(prob_list)-subtract_log)**2)-renyi_fast**2
            
        
        
    return renyi_fast_list,magic_capacity


def project_first_qubit(state):
    """
    measures first qubit of state 
    in computational basis and returns
    normalized projected state for each outcome
    and associated probabilitites
    
    """
    
    n_qubits=len(state.dims[0])

    n_qubits_remain=n_qubits-1

    if(n_qubits>1):
        dims_vec_remain=[[2]*(n_qubits_remain),[1]*(n_qubits_remain)]
    else:
        dims_vec_remain=[[1],[1]]
    amplitudes=state.data.toarray()[:,0]
    amplitudes0=amplitudes[:2**n_qubits_remain]
    amplitudes1=amplitudes[2**n_qubits_remain:]

    proj_prob0=np.sum(np.abs(amplitudes0)**2)
    proj_prob1=np.sum(np.abs(amplitudes1)**2)

    if(proj_prob0>0):
        proj_state0=qt.Qobj(amplitudes0/np.sqrt(proj_prob0),dims=dims_vec_remain)
    else:
        proj_state0=qt.Qobj(amplitudes0,dims=dims_vec_remain)
    
    if(proj_prob1>0):
        proj_state1=qt.Qobj(amplitudes1/np.sqrt(proj_prob1),dims=dims_vec_remain)
    else:
        proj_state1=qt.Qobj(amplitudes1,dims=dims_vec_remain)

    return proj_state0,proj_state1,proj_prob0,proj_prob1





def get_overlaps(state,max_qubit):
    """
    computes needed overlaps E for Von Neumann SE sampling in Feynmann trajectories
    Input:
        state 
        max_qubit, which is maximal number of projected qubits to compute overlaps for

    
    Output:
        overlaps_list: Overlaps computed
        probs_projected: Probability of projection
        states_projected: normalized state after projection
        
    """


    n_qubits=len(state.dims[0])

    
    overlaps_list=[]
    
    ##initial stuff
    states_projected=[state] ##start with initial state
    probs_projected=[1]

    ##go through qubits up to max_qubit, where we stop
    for order in range(n_qubits):
    
        n_qubits_left=len(states_projected[0].dims[0])
        
        if(n_qubits_left<=max_qubit):
            break
        else:
            new_states_projected=[]
            new_probs_projected=[]

            ##projet one additional qubit and add them to the list of projected states
            for k in range(len(states_projected)):
                curr_state=states_projected[k]
                curr_prob=probs_projected[k]
                
                ##sample left-most qubit
                proj_state0,proj_state1,proj_prob0,proj_prob1=project_first_qubit(curr_state)
        
                    
                new_states_projected.append(proj_state0)
                new_states_projected.append(proj_state1)
                new_probs_projected.append(curr_prob*proj_prob0)
                new_probs_projected.append(curr_prob*proj_prob1)



            states_projected=new_states_projected
            probs_projected=np.array(new_probs_projected)
            n_states=len(new_states_projected)
            

            overlaps=np.ones([n_states,n_states],dtype=np.complex128)
            
            states_array=np.array(states_projected)[:,:,0]
            ##compute overlaps between projected states
            for q in range(len(states_projected)):
                overlaps[:,q]=np.sqrt(probs_projected*probs_projected[q])*np.dot(states_array,np.conj(states_array[q]))
                
            overlaps_list.append(overlaps)



    return overlaps_list,probs_projected,states_projected



def sample_Pauli(state,max_qubit,rng,overlaps_list,probs_projected,states_projected,do_bell_sampling=False):
    """
    Performs Pauli sampling P\sim <psi|P|psi>^2
    
    Set do_bell_sampling=True to do Bell sampling instead which is P\sim <psi^*|P|psi>^2
    
    Main issue is that Pauli sampling requires too large memory to be done directly
    
    Thus, first performs Feynman like trajectory sampling which requires less memory
    then after projecting sufficient qubits
    switches to direct sampling from tensored state
    
    """
    
    n_qubits=len(state.dims[0])


    ##how bell transform is changing computational basis state 00, 01,10,11 
    ##first index: bell outcome, second index: components for that state, third index: which states are used from first and second copy
    bell_transform=np.array([[[0,0],[1,1]],[[0,1],[1,0]],[[0,0],[1,1]],[[0,1],[1,0]]],dtype=int) 
    bell_factor=1/np.sqrt(2)*np.array([[1,1],[1,1],[1,-1],[1,-1]]) ##amplitude factors
    
    

    bell_description1=[[]]
    bell_description2=[[]]
    bell_description_factor=[1]
       
    
    ##more than 12 is bad due to memory
    #max_qubit=min((n_qubits-4)//2,12) ##when to switch to constructing full vector
    
    prod_probs=1 ##initial probability
    log_prod_probs=0 ##logarithmic version due to floating point errors possible for high N
    bell_outcome=[] ##collect outcomes for bell sampling

    for order in range(n_qubits):
        ##go through qubits
        n_qubits_left=n_qubits-order
        
        if(n_qubits_left<=max_qubit):
            ##here we switch to direct sampling from full state


            if(len(bell_description_factor)==1):
                ##directly construct state if no trajectories have been done
                if(do_bell_sampling==False):##Pauli sampling
                    tensor_state=qt.tensor([state.conj(),state])
                else: ##Bell sampling
                    tensor_state=qt.tensor([state,state])
            else:
                ##construct full state from trajectories
                index_states=(2**np.arange(order))[::-1]
                tensor_state=0
                for k in range(len(bell_description1)):
                    ##go through superposition state which was generated
                    #print(k)
                    index1=sum(bell_description1[k]*index_states)
                    index2=sum(bell_description2[k]*index_states)
                    if(do_bell_sampling==False):##Pauli sampling
                        tensor_state+=bell_description_factor[k]*np.sqrt(probs_projected[index1])*np.sqrt(probs_projected[index2])*qt.tensor([states_projected[index1].conj(),states_projected[index2]])
                    else: ##Bell sampling
                        tensor_state+=bell_description_factor[k]*np.sqrt(probs_projected[index1])*np.sqrt(probs_projected[index2])*qt.tensor([states_projected[index1],states_projected[index2]])

            ##marginal sampling from full state
            
            hadamard_gate=qt.qip.operations.hadamard_transform(1)
            for i in range(n_qubits_left):
                ##go throuhg qubits left
                #print(i)
                n_qubits_now=n_qubits_left-i
                #print("Projecting on full state with remaining",n_qubits_now)
    
                
                ##permute order of qubits such that first qubit of each copy is next to each other at front
                tensor_state=tensor_state.permute([0,n_qubits_now]+list(np.arange(1,n_qubits_now))+list(np.arange(n_qubits_now+1,2*n_qubits_now)))
    
                ##perform Bell transformation
                tensor_state=qt.qip.operations.gate_expand_1toN(hadamard_gate, 2*n_qubits_now, 0)*qt.qip.operations.cnot(2*n_qubits_now,0,1)*tensor_state
                
                ##get projection of first qubit from Bell pair
                proj_state0,proj_state1,proj_prob0,proj_prob1=project_first_qubit(tensor_state)
    
                ##sample by computing probabilities of projection of each qubit
                ###and sampling from probabilities
                rand_val=rng.random()
                bell_index=0
                bell_prob=1
                if(proj_prob0>rand_val):
                    bell_prob*=proj_prob0
                    bell_index+=0
                    tensor_state=proj_state0
                else:
                    bell_prob*=proj_prob1
                    bell_index+=2
                    tensor_state=proj_state1
            
                ##second qubit of Bell pair
                proj_state0,proj_state1,proj_prob0,proj_prob1=project_first_qubit(tensor_state)
            
                rand_val=rng.random()
                if(proj_prob0>rand_val):
                    bell_prob*=proj_prob0
                    bell_index+=0
                    tensor_state=proj_state0
                else:
                    bell_prob*=proj_prob1
                    bell_index+=1
                    tensor_state=proj_state1
                
                bell_outcome.append(bell_index)
                prod_probs*=bell_prob
                log_prod_probs+=np.log(bell_prob)
                

            break #finished
                
        else:

            ##Feynman-like sampling from trajectories

            ##this is list of overlaps we already pre-computed
            overlaps=overlaps_list[order]
            
            ##numerates all possible states
            index_states=(2**np.arange(order+1))[::-1]
            

            sum_bell_prob=0
            
            rand_val=rng.random() ##sample random number for sampling
            
            ##compute probabilities of all bell states and sample
            for bell_index in range(4):
                ##is a description of trajectory of sampled outcomes
                ##new_bell_description1 describes first copy in tensor product
                ##new_bell_description2 describes second copy in tensor product
                new_bell_description1=[]
                new_bell_description2=[]
                
                bell_state_index1=np.zeros(2*len(bell_description_factor),dtype=int)
                bell_state_index2=np.zeros(2*len(bell_description_factor),dtype=int)
                new_bell_description_factor=[]
                
                ##go through all already projected states, and append new ones
                for k in range(len(bell_description_factor)):
                    ##first state
                    new_bell_description1.append(bell_description1[k]+[bell_transform[bell_index][0][0]]) ##first copy
                    new_bell_description2.append(bell_description2[k]+[bell_transform[bell_index][0][1]]) ##second copy
                    
                    bell_state_index1[2*k]=sum(new_bell_description1[2*k]*index_states)
                    bell_state_index2[2*k]=sum(new_bell_description2[2*k]*index_states)
    
                    ##second state
                    new_bell_description1.append(bell_description1[k]+[bell_transform[bell_index][1][0]]) ##first copy
                    new_bell_description2.append(bell_description2[k]+[bell_transform[bell_index][1][1]]) ##second copy
                    
                    bell_state_index1[2*k+1]=sum(new_bell_description1[2*k+1]*index_states)
                    bell_state_index2[2*k+1]=sum(new_bell_description2[2*k+1]*index_states)
                    
                    
                    new_bell_description_factor.append(bell_description_factor[k]*bell_factor[bell_index][0])
                    new_bell_description_factor.append(bell_description_factor[k]*bell_factor[bell_index][1])
                    
                    
                ##probability of sampling this outcome
                bell_prob=0
                ##get probability by going through all superposition states
                new_bell_description_factor=np.array(new_bell_description_factor)
                for k in range(len(new_bell_description_factor)):
                    if(do_bell_sampling==False):##Pauli sampling
                        bell_prob+=np.sum(new_bell_description_factor[k]*new_bell_description_factor*np.conjugate(overlaps[bell_state_index1[k],bell_state_index1])*overlaps[bell_state_index2[k],bell_state_index2])
                    else:
                        bell_prob+=np.sum(new_bell_description_factor[k]*new_bell_description_factor*overlaps[bell_state_index1[k],bell_state_index1]*overlaps[bell_state_index2[k],bell_state_index2])


                #print(bell_prob)
                bell_prob=np.real(bell_prob)
                sum_bell_prob+=bell_prob ##keep track of probabilities from previous outcomes

            
                if(sum_bell_prob>rand_val):
                    
                    ##select this path going forward
                    
                    ##keep description of states which were sampled
                    bell_description1=new_bell_description1
                    bell_description2=new_bell_description2
                    ##stores renormalization with sampling probability
                    bell_description_factor=np.array(new_bell_description_factor)/np.sqrt(bell_prob)
                    

                    ##store outcome and probabilities
                    bell_outcome.append(bell_index)
                    prod_probs*=bell_prob
                    log_prod_probs+=np.log(bell_prob)
        
                    break
        
                if(bell_index>=3):
                    ##check if something went wrong
                    raise NameError("Added Probabilities smaller than 1")
        
        
    map_outcome_pauli=np.array([0,1,3,2],dtype=int) ##map bell outcome to pauli operator
    
    ##expectation value of sampled pauli
    expect_pauli_sq=2**(n_qubits)*prod_probs ## is expectation value of pauli **2
    
    ##collect paulis sampled
    pauli_outcome=[map_outcome_pauli[bell_outcome[i]] for i in range(n_qubits)]

    return expect_pauli_sq,pauli_outcome


def prod(factors):
    return reduce(operator.mul, factors, 1)


def get_circuit_T(n_qubits,depth,n_nonclifford):
    ##Circuit with Clifford operations and T-gates#
    ##uses depth layers of CNOT+local rotations, chosen as Clifford
    ##n_nonclifford determines number of non-Clifford single-qubit gates, which are chosen as T-gates (or some Clifford rotated version)
    n_circuit_parameters=2*depth*n_qubits
    

    circuit_params=rng.integers(0,4,n_circuit_parameters)*np.pi/2
    which_params=np.arange(n_circuit_parameters)
    rng.shuffle(which_params)

    for i in range(min(n_circuit_parameters,n_nonclifford)):
        circuit_params[which_params[i]]=rng.integers(0,4)*np.pi/2+np.pi/4


    
    
    if(n_qubits==2):
        entangling_gate_index=[[0,1]]
    else:
        entangling_gate_index=[[2*j,(2*j+1)%n_qubits] for j in range(int(np.ceil(n_qubits/2)))]+[[2*j+1,(2*j+2)%n_qubits] for j in range((n_qubits)//2)]
    
    if(n_qubits>1):
        opEntangler=[prod([qt.qip.operations.cnot(n_qubits,j,k) for j,k in entangling_gate_index[::-1]])]
    else:
        opId=genFockOp(qt.qeye(2),0,n_qubits)
        opEntangler=[opId]
        

    circuit_state=qt.tensor([qt.basis(2,0) for i in range(n_qubits)])


        
    for i in range(depth):
        for j in range(n_qubits):
            param_index=i*n_qubits*2+j
            angle=circuit_params[param_index]
            rot_op=qt.qip.operations.ry(angle,n_qubits,j)
            circuit_state=rot_op*circuit_state
            

                    
        for j in range(n_qubits):
            param_index=i*n_qubits*2+n_qubits+j
            angle=circuit_params[param_index]
            
            rot_op=qt.qip.operations.rz(angle,n_qubits,j)
            circuit_state=rot_op*circuit_state
            

        circuit_state=opEntangler[i%len(opEntangler)]*circuit_state

    return circuit_state



def standard_normal_complex(size):
    """return ``(R + 1.j*I)`` for independent `R` and `I` from np.random.standard_normal."""
    return np.random.standard_normal(size) + 1.j * np.random.standard_normal(size)


def GUE(size):
    r"""Gaussian unitary ensemble (GUE).
    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.
    Returns
    -------
    H : ndarray
        Hermitian (complex) numpy matrix drawn from the GUE, i.e.
        :math:`p(H) = 1/Z exp(-n/4 tr(H^2))`.
    """
    A = standard_normal_complex(size)
    return (A + A.T.conj()) * 0.5



rng=np.random.default_rng()




dims_vec=[[2]*n_qubits,[1]*n_qubits]
dims_mat=[[2]*n_qubits,[2]*n_qubits]


##type of state to compute magic from
if(model==0):
    ##0 state
    state=qt.tensor([qt.basis(2,0) for i in range(n_qubits)])
elif(model==1):
    ##T state
    state=qt.tensor([qt.basis(2,0)+np.exp(1j*np.pi/4)*qt.basis(2,1) for i in range(n_qubits)])
elif(model==2):
    ##haar random state
    
    dims_vec=[[2]*(n_qubits),[1]*(n_qubits)]
    # state=qt.rand_ket_haar(N=2**(n_qubits),dims=dims_vec)
    
    state=qt.Qobj(standard_normal_complex(2**n_qubits),dims=dims_vec)
    state=state/state.norm()
    
elif(model==3): ##GUE
    GUE_matrix=2**(-n_qubits/2)*qt.Qobj(GUE([2**n_qubits,2**n_qubits]),dims=dims_mat) #this nearly normalizes GUE matrix according to psi H psi^2
    GUE_matrix=GUE_matrix/np.sqrt((GUE_matrix.dag()*GUE_matrix).tr()/(2**(n_qubits)+1)) ##+1 factor comes from haar random integration
    #norm=(GUE_matrix.dag()*GUE_matrix).tr()/(2**(n_qubits)+1)
    ###norm=(GUE_matrix.dag()*GUE_matrix).tr()*2**(-n_qubits)
    state0=qt.tensor([qt.basis(2,0) for i in range(n_qubits)])
    state0=state0/state0.norm()
    state=qt.Qobj(state0)
    state=(-1j*evolution_time*GUE_matrix).expm()*state
elif(model==4):## Clifford +T
    state=get_circuit_T(n_qubits, layers, n_tgates)

state=state/state.norm()


print("start")

expect_pauli_sq_list=[]

print("Getting overlaps")
##precompute overlaps needed for Von Neumann sampling
overlaps_list,probs_projected,states_projected=get_overlaps(state,max_qubit)
print("Got overlaps")


sampling_time=time.time()
print("Start sampling")
percent=n_sample//10
for step in range(n_sample):
    if(percent>0):
      if((step+1)%percent==0):
        print("Done",step//percent*10,"%")

    expect_pauli_sq,pauli_outcome=sample_Pauli(state,max_qubit,rng,overlaps_list,probs_projected,states_projected,do_bell_sampling=do_bell_sampling)
    expect_pauli_sq_list.append(expect_pauli_sq)

    

print("Finished sampling")
expect_pauli_sq_list=np.array(expect_pauli_sq_list)

##sampling estimation of von neumann SE density
vn_sample=np.mean(-np.log(2**(-n_qubits)*expect_pauli_sq_list)-n_qubits*np.log(2))/n_qubits

##magic capacity
magic_capacity=np.var(-np.log(2**(-n_qubits)*expect_pauli_sq_list)-n_qubits*np.log(2))
vn_sample_std=np.sqrt(magic_capacity)/np.sqrt(n_sample)/n_qubits ##standard deviation of von neummann SRE density estimator


print("Magic capacity",magic_capacity)

print("Von Neumann SE density",vn_sample,"+-",vn_sample_std)



print("Sampling time",time.time()-sampling_time)

if(n_qubits<=8): ##test against exact SE, slow method
    exact_slow_time=time.time()
    ##get all pauli expectation values
    pauli_op_list,pauli_list=get_all_paulis_N(n_qubits)
    pauli_sq_expect_exact=np.array([qt.expect(pauli_op_list[k],state)**2 for k in range(len(pauli_op_list))])
    
    pauli_sq_expect_nonzero=pauli_sq_expect_exact[pauli_sq_expect_exact>0]
    vn_exact=-2**-n_qubits*np.sum(pauli_sq_expect_nonzero*np.log(pauli_sq_expect_nonzero))/n_qubits
    
    print("Von Neumann SE density exact",vn_exact)
    
    index_pauli=np.argmax(np.all(pauli_list==pauli_outcome,axis=1))
    if(do_bell_sampling==False):
        ##check that the last sampled pauli had correct probability
        ##pauli sampling
        ##P\sim <psi|P|psi>^2
        print("Compare one sampled outcome with exact value",pauli_sq_expect_exact[index_pauli],expect_pauli_sq,pauli_outcome)
    else:
        ##bell sampling
        ##P\sim <psi^*|P|psi>^2
        bell_prob_exact=np.abs((state.dag().conj()*pauli_op_list[index_pauli]*state).tr())**2
        print("Compare one Bell sampled outcome with exact value",bell_prob_exact,expect_pauli_sq,pauli_outcome)

        
    print("Exact slow time",time.time()-exact_slow_time)

##creates basis needed to compute stabilizer entropy for up to 13 qubits
##faster method
if(n_qubits<=13):
    exact_fast_time=time.time()
    base_states=np.array([numberToBase(i, 2,n_qubits) for i in range(2**n_qubits)],dtype=int)

    ##these steps need only be run once
    print("Get magic bases")
    conversion_matrix_binary_prod=get_conversion_matrix_binary_prod(base_states)
    print("Get mod add base")
    conversion_matrix_mod_add_index=get_conversion_matrix_mod_add_index(base_states)
    print("Finish magic bases")

    ##compute stabilizer entropy, any alpha can be chosen
    renyi_fast_list,magic_capacity_exact=renyi_entropy_fast(state,conversion_matrix_mod_add_index,conversion_matrix_binary_prod,alpha=[1])
    print("SE density exact with fast method",np.array(renyi_fast_list)/n_qubits)
    print("magic capacity with exact method",magic_capacity_exact)
    
    print("Exact fast time",time.time()-exact_fast_time)