# Copyright 2024, Nicholas Lotz

# This file is part of Oblic.
# Oblic is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Oblic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License along with Oblic. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# Function to convert qubit amplitudes to a column vector representation
def qubit_to_column_vector(alpha, beta):
    # Create and return a 2x1 numpy array representing the qubit state
    return np.array([[alpha], [beta]])

# Function to calculate the probabilities of a qubit being in each state
def calculate_probabilities(column_vector):
    # Square the absolute values of the elements to get probabilities
    return np.abs(column_vector)**2 

# Class representing a ket vector (standard quantum state vector)
class ket:
    # Initialize the ket vector with amplitudes for the |0⟩ and |1⟩ states
    def __init__(self, alpha, beta):
        # Store the column vector representing the quantum state
        self.column_vector = qubit_to_column_vector(alpha, beta)
        # Calculate and store the probabilities of the state
        self.probabilities = calculate_probabilities(self.column_vector)
        # Calculate and store the conjugate transpose (bra vector) of the ket
        self.conjugate_transpose = self.column_vector.conj().T

# Class representing a bra vector (conjugate transpose of a ket vector)
class bra:
    # Initialize the bra vector based on a given ket vector
    def __init__(self, ket):
        # Store the row vector which is the conjugate transpose of the ket's column vector
        self.row_vector = ket.conjugate_transpose
        # Calculate the probabilities of the state based on the row vector
        self.probabilities = calculate_probabilities(self.row_vector.T)  # T converts it back to column for calculation

# Function to convert a ket vector to a bra vector
def convert_ket_to_bra(ket):
    # Create and return a bra object based on the provided ket
    return bra(ket)

# Function to convert a bra vector back to a ket vector
def convert_bra_to_ket(bra):
    # Calculate the original amplitudes by taking conjugate transpose of the bra and convert it to a ket
    return ket(bra.row_vector.conj().T[0][0], bra.row_vector.conj().T[1][0])

# Calculate the inner product of a bra and a ket vector
def inner_product(state1, state2, tolerance=1e-10):
    bra_state1 = convert_ket_to_bra(state1)
    result = np.dot(bra_state1.row_vector, state2.column_vector)[0][0]
    if abs(result) < tolerance:
        return 0
    else:
        return result

# Determine if state (assume ket) is normalized
def is_normalized(state, tolerance=1e-10):
    # Calculate the inner product of the state with itself
    norm = inner_product(state, state)
    # Check if the absolute difference from 1 is within the tolerance
    if abs(norm - 1) < tolerance:
        return True
    else:
        return False

# Determine if states are orthogonal
# state1 must be bra, state2 must be ket
def is_orthogonal(state1, state2):
    if inner_product(state1, state2) == 0:
        return True
    else:
        return False

# States are orthonormal if normalized and orthogonal
def is_orthonormal(state1, state2):
    if (is_normalized(state1) and is_normalized(state2) and is_orthogonal(state1, state2)):
        return True
    else:
        return False


# EXAMPLES

# Example complex ket and bra
# example_ket = ket(((3+1.73205080757j)/4), (1/2))
# example_bra = convert_ket_to_bra(ket((1/4), (3.87298334621/4))) 

# Output inner product of example vectors
# print("Inner product is: ", inner_product(example_bra, example_ket))

# Determine if normalized
# example_state = ket(2, 3j)
# print("Inner product is: ", inner_product(convert_ket_to_bra(example_state), example_state))
# print("Is normalized?", is_normalized(example_state))
# coeff = 1/np.sqrt(13)
# example_state = ket(coeff*2, coeff*3j)
# print("Inner product is: ", inner_product(convert_ket_to_bra(example_state), example_state))
# print("Is normalized?", is_normalized(example_state))


# Determine if orthogonal - example 1
# minus_ket = ket(1/np.sqrt(2), 1/np.sqrt(2))
# plus_ket = ket(1/np.sqrt(2), -1/np.sqrt(2))
# print("Inner product is: ", inner_product(minus_ket, plus_ket))
# print("Is orthogonal?", is_orthogonal(minus_ket, plus_ket))

# Determine if orthogonal - example 2
# zero_ket = ket(1, 0)
# plus_ket = ket(1/np.sqrt(2), -1/np.sqrt(2))
# print("Inner product is: ", inner_product(zero_ket, plus_ket))
# print("Is orthogonal?", is_orthogonal(zero_ket, plus_ket))

# Determine if orthogonal - example 3
# state1 = ket((1 + np.sqrt(3)*1j)/4, (np.sqrt(2) - 1j)/2)
# state2 = ket((np.sqrt(2) + 1j)/2, ((-1 + np.sqrt(3)*1j)/4)) 
# print("Inner product is: ", inner_product(state1, state2))
# print("Is orthogonal?", is_orthogonal(state1, state2))

# Determine if orthonormal - example 1
minus_ket = ket(1/np.sqrt(2), 1/np.sqrt(2))
plus_ket = ket(1/np.sqrt(2), -1/np.sqrt(2))
print("Inner product of state1 is ", inner_product(minus_ket, minus_ket))
print("Inner product of state2 is ", inner_product(plus_ket, plus_ket))
print("Inner product of state1 and state2: ", inner_product(minus_ket, plus_ket))
print("Is orthogonal?", is_orthogonal(minus_ket, plus_ket))
print("State1 normalized?", is_normalized(minus_ket))
print("State2 normalized?", is_normalized(plus_ket))
print("Is orthonormal?", is_orthonormal(minus_ket, plus_ket))
