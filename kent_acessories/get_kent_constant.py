from scipy.special import iv as I_, gamma as G_
import numpy as np

def __c(kappa, beta, terms):
    """
    Calculates the normalization constant for the Kent (FB5) distribution.

    Args:
        kappa: Concentration parameter.
        beta: Ovalness parameter.
        terms: Number of terms to include in the series approximation.

    Returns:
        The normalization constant.
    """
    su = 0
    for j in range(0, terms):
        su += G_(j + .5) / G_(j + 1) * beta**(2*j) * (2/kappa)**(2*j + .5) * I_(2*j + .5, kappa)
        #print(j, I_(2*j+0.5, kappa))

    #print(f'Primeiro termo: {2*np.pi*G_(.5) / G_(1) * (2/kappa)**(.5) * I_(.5, kappa)}')
    return 2 * np.pi * su

def c_approx(kappa, beta):
    """Corrected approximation for the normalization constant c(kappa, beta)."""
    return 2*np.pi*np.exp(kappa)*((kappa - 2 * beta)*(kappa + 2 * beta))**(-0.5)
    #return (kappa - 2 * beta) / (kappa + 2 * beta)
    #return (((kappa - 2 * beta)*(kappa + 2 * beta))**(0.5))/2*np.pi*np.exp(kappa)

def evaluate_constant(kappa, beta, max_iterations=1000):
    """
    Evaluates the constant __c with increasing precision and compares with approximation.

    Args:
        kappa: Concentration parameter.
        beta: Ovalness pacdrameter.
        max_iterations: Maximum number of iterations to evaluate.

    Returns:
        A list of tuples (iterations, constant_value_series, constant_value_approx).
    """
    results = []
    previous_value = 0  
    for iterations in range(1, max_iterations + 1):
        constant_value_series = __c(kappa, beta, terms=iterations)
        constant_value_approx = c_approx(kappa, beta)
        results.append((iterations, constant_value_series, constant_value_approx))
        if abs(constant_value_series - previous_value) < 1e-8:  # Check for convergence
            break 
        previous_value = constant_value_series
    return results

# Example usage with fixed kappa and beta
kappa, beta =   1.6911, 0.6440
results = evaluate_constant(kappa, beta)

# Print results in a table
print("Iterations | Series Approximation | Approx. Formula | Difference   ")
print("-----------|--------------------|----------------|------------")
for iterations, c_series, c_approx_value in results:
    print(f"{iterations:10} | {c_series:.8f}        | {c_approx_value:.8f} | {(c_series - c_approx_value) / c_approx_value:.8f}")