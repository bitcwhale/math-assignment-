
import numpy as np
import pandas as pd
import sympy as sp
from scipy.linalg import null_space
# Question 1__________________________________________
print("Question 1")
# Data:
expected_returns = np.array([5.8, 7.6, 8.3, 5.8])
weights = np.array([0.5, 0, 0, 0.5])
covmatrix = np.array([[3.21, -3.52, 6.99, 0.04],
                      [-3.52, 5.84, -13.68, 0.12],
                      [6.99, -13.68, 61.81, -1.64],
                      [0.04, 0.12, -1.64, 0.36]])

# E[Rp]
expected_portfolio_return = np.dot(weights, expected_returns)
print("expected_portfolio_return =", expected_portfolio_return)

# Variance of the portfolio using the formula v=x'TCovmatrix
# Variance calculation
portfolio_variance = np.dot(weights.T, np.dot(covmatrix, weights))
print("portfolio_variance =", portfolio_variance)

# Asset 1 return and risk
A1return = expected_returns[0]
A1variance = covmatrix[0, 0]
print("Asset 1 return is equal to", A1return)
print("Asset 1 variance is equal to", A1variance)

# Asset 4 return and risk
A4return = expected_returns[3]
A4variance = covmatrix[3, 3]
print("Asset 4 return is equal to", A4return)
print("Asset 4 variance is equal to", A4variance)

# Comparison
if expected_portfolio_return > max(A1return, A4return):
    print("The diversified portfolio has a higher expected return than either individual asset.")
else:
    print("The diversified portfolio does not have a higher expected return than either individual asset.")

if portfolio_variance < min(A1variance, A4variance):
    print("The diversified portfolio has a lower variance than either individual asset.")
else:
    print("The diversified portfolio does not have a lower variance than either individual asset.")

# Question 2__________________________________________
print("Question 2:")
# Data
Return = np.array([5.8, 7.6, 8.3, 5.8])
# Define symbolic variables
w = sp.symbols('w1 w2 w3 w4')
# Create expected return expression
expected_return = np.dot(Return, w)
print("expected_return =", expected_return)
# question 3________________________________________
print("Question 3")

E = np.array([[3.21, -3.52, 6.99, 0.04],
              [-3.52, 5.84, -13.68, 0.12],
              [6.99, -13.68, 61.81, -1.64],
              [0.04, 0.12, -1.64, 0.36]])

# symmetric
def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)

# orthogonal
def is_orthogonal(matrix):
    identity_matrix = np.eye(matrix.shape[0])
    return np.allclose(np.dot(matrix.T, matrix), identity_matrix)

# idempotent
def is_idempotent(matrix):
    return np.array_equal(np.dot(matrix, matrix), matrix)


print("Is symmetric:", is_symmetric(E))
print("Is orthogonal:", is_orthogonal(E))
print("Is idempotent:", is_idempotent(E))

#question 4________________________________
print("Question 4")
# here I use the same covariance matrix as the previous one did not change
variances = np.diag(E)

# calcul the minimum variance with only one asset invested in
min_variance_place = np.where(variances == np.min(variances))[0][0]

# now what is the max weight i could get with the minimum variance in cash otherwise
optimal_weight = np.zeros(len(variances))
optimal_weight[min_variance_place] = 1

# Portfolio variance when investing in the asset with minimum variance
optimal_variance = variances[min_variance_place]

# results
print(f"The optimal asset to invest in, is Asset: {min_variance_place+1}") #I had +1 to represent the good asset in the print
print(f"The optimal portfolio weights: {optimal_weight}")
print(f"The portfolio variance: {optimal_variance}")

#question 5_____________________________________________
print("Question 5")
#determinant
print("The determinant of matrix E is")
detE=np.linalg.det(E).round()
print(detE)
#trace
print("The trace of matrix E is")
trE=np.trace(E)
print(trE)
#inverse
print("The inverse of matrix E is")
invE = pd.DataFrame(np.linalg.inv(E))
print(invE)
#adjoint matrix
print("The adjoint of Matrix E is")
adjE = detE*invE
print(adjE)
#kernel of the matrix
print ("the kernel of the matrix E =")
kernel_E = null_space(E)
print(kernel_E)
# Check if the kernel is empty
if kernel_E.size == 0:
    print("The kernel is trivial (only the zero vector).")

#question 6__________________________________________
print("Question 6")
#restricted the number of decimals
eigenvalues = np.linalg.eig(E)[0]
print("the eigenvalues:")
print(np.round(eigenvalues,4))

# We put the eigenvalues in the diagonal matrix Z
eigenvalues_Z =pd.DataFrame(np.diag(eigenvalues))
print("the eiganvalues in Z : ")
print(eigenvalues_Z)

# We compute the eigenvectors matrix P
eigenvectors =pd.DataFrame(np.linalg.eig(E)[1])
print("eigenvectors :")
print(eigenvectors)
#question 7__________________________________________
print("Question 7")
if np.all(eigenvalues > 0):
    print("E is Positive Definite")
elif np.all(eigenvalues >= 0):
    print("E is Positive Semidefinite")
elif np.all(eigenvalues < 0):
    print("E is Negative Definite")
elif np.all(eigenvalues <= 0):
    print("E is Negative Semidefinite")
else:
    print("E is Indefinite")
#question 8_________________________________________
print("Question 8")
#let's recompute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(E)


#here we check if the eigenvalue are non zero otherwise the matrix is not invertible but singular
if np.any(np.isclose(eigenvalues, 0)):
    print("The matrix E is not invertible (has zero eigenvalue).")
else:
    # Create diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)

    # Compute the inverse of E using spectral decomposition
    # Inverse is V * D^-1 * V^T
    D_inv = np.diag(1 / eigenvalues)  # Inverse of the diagonal matrix of eigenvalues
    E_inv_spectral = eigenvectors @ D_inv @ eigenvectors.T #here is the computation of the inverse of the E matrix

    # Display the inverse from spectral decomposition
    print("Inverse of E using Spectral Decomposition:")
    print(E_inv_spectral)


# Compute Cholesky decomposition
# the matrix must be positive definite to be computed this way
try:
    L = np.linalg.cholesky(E)
    print("Cholesky Decomposition (L):")
    print(L)
    # verification to check if L * L^T equals E
    E_reconstructed = L @ L.T  # Check if it's equal to E
    print("Reconstructed E from Cholesky Decomposition:")
    print(E_reconstructed)

# Cholesky decomposition fails
except np.linalg.LinAlgError:
    print("Cholesky decomposition failed. The matrix may not be positive definite.")
#question 9 _______________________________
print("Question 9")
#let's define our matrix E
E = np.array([[3.21, -3.52, 6.99, 0.04],
              [-3.52, 5.84, -13.68, 0.12],
              [6.99, -13.68, 61.81, -1.64],
              [0.04, 0.12, -1.64, 0.36]])

# Vector of ones
ones = np.ones(E.shape[0])

# Compute the inverse of E

E_inv = np.linalg.inv(E)

# Compute E^(-1) * ones
numerator = E_inv @ ones

# Compute 1^T * E^(-1) * ones
denominator = ones.T @ E_inv @ ones

# Compute the MV portfolio weights
w_MV = numerator / denominator
print("Minimum Variance Portfolio Weights (MV):")
print(w_MV)

# Compute the variance of the MV portfolio
variance_MV = 1 / denominator
print("Variance of the Minimum Variance Portfolio:")
print(variance_MV)
#question 10 _______________________________________
print("Question 10")
mu = np.array([5.8, 7.6, 8.3, 5.8])
rf=0

# Compute E^(-1) *(mu-rf * ones)
numerator_new = E_inv @(mu-rf*ones)

# Compute 1^T * E^(-1) * (mu-rf * ones)
denominator_new = ones.T @ E_inv @ (mu-rf*ones)

# Compute the MV portfolio weights
new_w_MV = numerator_new / denominator_new
print("the NEW Minimum Variance Portfolio Weights (MV):")
print(new_w_MV)

# Compute the variance of the MV portfolio
new_variance_MV = 1 / denominator_new
print("The NEW Variance of the Minimum Variance Portfolio:")
print(new_variance_MV)

#exercice 4 part 5___________________________________________

# Define matrix A using NumPy
A = np.array([[2, 0, 1, 1],
              [-2, 2, 1, -1],
              [0, 0, 3, 1],
              [0, -2, -1, 3]])

#(b) calculate the A^(-1) matrix using adjoint method
detA=np.linalg.det(A).round()
invA = pd.DataFrame(np.linalg.inv(A)) #calculate the inverse
adjA = detA*invA
adjA_df = pd.DataFrame(adjA)
print("The adjoint of Matrix A is:")
print(adjA_df)

#(c)  x = b^t*A^(-1)

b=np.array([1,1,1,1])
x=invA@b.T
print ("the answer to the equation Ax=b is x equal :")
print(x)

# Step 1: Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Step 2: Clean eigenvalues and eigenvectors for readability
eigenvalues_rounded = np.round(eigenvalues, decimals=4)
eigenvectors_rounded = np.round(eigenvectors, decimals=4)

# Step 3: Remove small imaginary parts (if any)
eigenvalues_cleaned = np.real_if_close(eigenvalues_rounded, tol=1e-10)
eigenvectors_cleaned = np.real_if_close(eigenvectors_rounded, tol=1e-10)

# Step 4: Check for singularity of the eigenvector matrix
if np.any(np.isclose(eigenvalues_cleaned, 0)):
    print("The matrix is not invertible (has zero eigenvalue). Using pseudo-inverse.")
    C_inv = np.linalg.inv(eigenvectors_cleaned)  # Use pseudo-inverse if singular
else:
    C_inv = np.linalg.inv(eigenvectors_cleaned)    # the inverse if not singular therefore impossible to calculate the inverse

# Step 5: Create diagonal matrix of eigenvalues (Lambda)
D = np.diag(eigenvalues_cleaned)

# Step 6: Convert NumPy arrays to SymPy matrices for symbolic representation
A_sym = sp.Matrix(A)
C_sym = sp.Matrix(eigenvectors_cleaned)
Lambda_sym = sp.Matrix(D)
C_inv_sym = sp.Matrix(C_inv)

# Step 7: Display the symbolic equation for eigenvalue decomposition A = C * Lambda * C_inv
print("Eigenvalue Decomposition Equation: A = C * Lambda * C_inv\n")
sp.pprint(sp.Eq(A_sym, C_sym * Lambda_sym * C_inv_sym))



