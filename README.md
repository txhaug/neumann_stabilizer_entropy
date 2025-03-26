# Statevector algorithm to perform Bell/Pauli sampling, compute SRE and magic capacity

Statevector algorithm to perform Bell/Pauli sampling, compute SRE and magic capacity for arbitrary states.
Has been tested for up to 24 qubits.

Companion code to "Efficient mutual magic, magic capacity, and magic witnesses", arXiv:xx by Tobias Haug and Poetri Sonya Tarabunga


Works by using Bell sampling to sample from Pauli spectrum P(\sigma)=2^-N <psi|\sigma|psi>^2.
Bell sampling is done by sampling in computational basis from U_bell^{\otimes N} |psi^*>\otimes|psi>
which is equivalent to sampling from P(\sigma), where U_bell is the Bell transformation.
von Neumann SE can be estimated as -\sum_\sigma P(\sigma) log(<psi|\sigma|psi>^2)

Main problem is that |psi^*>\otimes|psi> is too large to be stored in memory.
Thus, first we use Feynman like approach where we store only |psi> and describe sampling trajectory.
of tensor product as a superposition (see manuscript for details).

This program uses this Feynmann-like trajectory approach to sample the first half of the qubits, 
then switches to direct sampling from marginals once enough qubits have been sampled.

This way, scaling of O((2\sqrt{2})^N) is achieved, tested up to 24 qubits.
For statevector simulation, best previous best method of O(8^N) which was limited to ~15 qubits.


NOTE: Requires older version of qutip, namely <=4.7.5, which has dependencies on older versions of numyp and scipy.
To install, make clean python 3.11 environment and install packages as:
- pip install numpy==1.26.4
- pip install scipy==1.12.0
- pip install qutip==4.7.5

- 
@author: Tobias Haug, 
Technology Innovation Institute, UAE
tobias.haug@u.nus.edu
