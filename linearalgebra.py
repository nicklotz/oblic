# Copyright 2024, Nicholas Lotz

# This file is part of Oblic.
# Oblic is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Oblic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License along with Oblic. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

# Alpha and beta are the amplitudes of the |0⟩ and |1⟩ states
def qubit_to_column_vector(alpha, beta): 
    # Default column vector for |0⟩ state
    ketzero_unit_column_vector = np.array([[1], [0]])  
    # Default column vector for |1⟩ state
    ketone_unit_column_vector = np.array([[0], [1]])  
    # Column vector is a linear combination of the |0⟩ and |1⟩ states
    return alpha*ketzero_unit_column_vector + beta*ketone_unit_column_vector

# Calculate probabilities of the qubit being in |0⟩ and |1⟩ states
def calculate_probabilities(column_vector):
    # Probabilities are the square of the absolute values of the amplitudes
    return np.abs(column_vector)**2 

# Calculate transpose of a vector by swapping rows and columns
def transpose_vector(vector):
    # Transpose of a column vector is a row vector
    return np.transpose(vector) 

# Calculate complex conjugate of a vector
def complex_conjugate(vector):
    # Complex conjugate of a column vector is a column vector with complex conjugate of the elements
    return np.conj(vector) 

# Calculate conjugate transpose of a vector
def conjugate_transpose(vector):
    # Conjugate transpose of a column vector is a row vector with complex conjugate of the elements
    return np.conj(np.transpose(vector)) 

# Ket is a column vector which is the default representation of a qubit
class ket:
    # Alpha and beta are the amplitudes of the |0⟩ and |1⟩ states
    def __init__(self, alpha, beta): 
        self.column_vector = qubit_to_column_vector(alpha, beta) 
        self.probabilities = calculate_probabilities(self.column_vector)
        self.transpose = transpose_vector(self.column_vector) 
        self.complex_conjugate = complex_conjugate(self.column_vector) 
        self.conjugate_transpose = conjugate_transpose(self.column_vector) 

# Bra is a row vector which is the conjugate transpose of a ket
class bra:
    def __init__(self, alpha, beta):
        self.column_vector = qubit_to_column_vector(alpha, beta) 
        self.probabilities = calculate_probabilities(self.column_vector)  
        self.transpose = transpose_vector(self.column_vector)
        self.complex_conjugate = complex_conjugate(self.column_vector) 
        self.conjugate_transpose = conjugate_transpose(self.column_vector) 

# Convert a ket to a bra by taking the conjugate transpose of the ket
def convert_ket_to_bra(ket):
    return bra(np.conj(ket.column_vector[0][0]), np.conj(ket.column_vector[1][0]))

# Convert a bra to a ket by taking the conjugate transpose of the bra
def convert_bra_to_ket(bra): 
    return ket(np.conj(bra.column_vector[0][0]), np.conj(bra.column_vector[1][0]))


# EXAMPLES

# Example complex ket and ra
example_complex_ket = ket((2/3), ((1/3)-(2j/3)))
example_complex_bra = convert_ket_to_bra(example_complex_ket) 

# Print the ket column vector
print("Ket is: ", example_complex_ket.column_vector) 

# Print the bra column vector after converting ket to bra
print("Bra after converting ket to bra is: ", example_complex_bra.column_vector)

