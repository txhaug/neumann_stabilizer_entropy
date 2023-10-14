# Universal dynamics of stabilizer entropy

Program to compute von Neumann Stabilizer entropy for arbitrary states
Tested up to 24 qubits
Works by using Bell sampling to sample from Pauli spectrum P(\sigma)=2^-N <psi|\sigma|psi>^2
Bell sampling is done by sampling in computational basis from U_bell^{\otimes N} |psi^*>\otimes|psi>
which is equivalent to sampling from P(\sigma), where U_bell is the Bell transformation.
Use this to estimate von Neumann SE -\sum_\sigma P(\sigma) log(<psi|\sigma|psi>^2)

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
