import numpy as np
import GPyOpt

#Defining experimental reference values
DSS_ref = 1
BG_ref = 3.2
v0_ref = 136.278
ref_3d= 1.24774
ref_2p= 4.66316

# Defining constants for normalising U, c1 and c2
U_min = 0.5
U_range = 2.5
HUBC1_min = 0.9
HUBC1_range = 0.2
HUBC2_min = -0.5
HUBC2_range = 0.5

# Defining constants for normalising tr_3d and tr_2p
tr_3d_min = 0.77094
tr_3d_range = 0.7449
tr_2p_min = 4.675545
tr_2p_range = 0.752689999999999

# Defining constants for SVM linear decision boundaries
A1 = -4.01
B1 = -1.14
C1 = -6.29
D1 = -5.16
A2 = -5.65
B2 = -4.07
C2 = 1.36
D2 = -4.65

# Defining constants for SISSO-derived emprical correlation for defect state separation from CBM
c0_DSS = 0.14693410613768
a0_DSS = 0.047077444290661
a1_DSS = 0.00916683216650025
a2_DSS = 0.295764153322452

# Defining constants for SISSO-derived emprical correlation for band gap
c0_BG = 2.53583356257976
a0_BG = -0.0478583432914388
a1_BG = -0.474335064859773
a2_BG = 0.0685378988282102

# Defining constants for SISSO-derived emprical correlation for Ti 3d occ. matrix trace
c0_tr_3d = 1.67456113459636
a0_tr_3d = -1.25867674751014
a1_tr_3d = -0.595870387849108

# Defining constants for SISSO-derived emprical correlation for O 2p occ. matrix trace
c0_tr_2p =5.29651194428234
a0_tr_2p = -0.424431720965629


def calculate_band_gap(U, HUBC1, HUBC2):
    term1 = (HUBC1 ** 6) * np.log(HUBC1)
    term2 = np.exp(U) * np.sin(HUBC2)
    term3 = (HUBC2 ** 6) * U
    term4 = (HUBC1 ** 6) - np.sin(HUBC1)
    term5 = np.exp((HUBC1 ** 3))
    term6 = U / HUBC1
    term7 = (HUBC2 ** 3)

    band_gap = c0_BG + a0_BG * (term1 * term2) + a1_BG * (term3 / term4) + a2_BG * (term5 * (term6 + term7))

    return band_gap

def calculate_DSS(U, HUBC1, HUBC2):
    term1 = (U ** 2) * np.exp(HUBC2)
    term2 = np.cbrt(HUBC1 - U)
    term3 = np.exp(HUBC2) / np.exp(U)
    term4 = (HUBC1 ** 6) - np.sin(HUBC1)
    term5 = np.exp(HUBC1) - (1.0 / U)
    term6 = (HUBC1 ** 3) - (1.0 / U)

    DSS = c0_DSS + a0_DSS * (term1 + term2) + a1_DSS * (term3 / term4) + a2_DSS * (term5 * term6)

    return DSS

def calculate_v0(U, HUBC1, HUBC2):
    normalized_U = (U - U_min) / U_range
    normalized_HUBC1 = (HUBC1 - HUBC1_min) / HUBC1_range
    normalized_HUBC2 = (HUBC2 - HUBC2_min) / HUBC2_range

    term1 = 0.789080849435898 * ((normalized_HUBC1**6) * (normalized_HUBC2 * normalized_U))
    term2 = 0.0910729975623938 * (abs((normalized_HUBC1 + normalized_U) - (normalized_HUBC2**6)))
    term3 = -0.723462573622041 * ((normalized_U**2) * (normalized_HUBC2**3))
    term4 = 0.755014182723015
    v0 = (term1 + term2 + term3 + term4) * 0.893167975915986 + 137.067036242666
    return v0

def calculate_tr_3d_norm(U, HUBC1, HUBC2):
    # Calculate each term of the tr_3d formula
    term1 = a0_tr_3d * (((1.0 / U) - (HUBC2 * U)) * (np.cos(HUBC1) ** 6))
    term2 = a1_tr_3d * (np.exp(HUBC2 * U) - (HUBC1 ** 3 - np.cbrt(U)))

    # Compute tr_3d
    tr_3d = c0_tr_3d + term1 + term2

    # Normalize tr_3d
    tr_3d_normalised = (tr_3d - tr_3d_min) / tr_3d_range

    return tr_3d_normalised


def calculate_tr_2p_norm(U, HUBC1, HUBC2):
    # Calculate each term of the tr_2p formula
    term1 = a0_tr_2p * (np.cos(np.sqrt(U)) - ((U / HUBC1) * np.sin(HUBC2)))

    # Compute tr_2p
    tr_2p = c0_tr_2p + term1

    # Normalize tr_2p
    tr_2p_normalised = (tr_2p - tr_2p_min) / tr_2p_range

    return tr_2p_normalised

def calculate_tr_3d(U, HUBC1, HUBC2):
    # Calculate each term of the tr_3d formula
    term1 = a0_tr_3d * (((1.0 / U) - (HUBC2 * U)) * (np.cos(HUBC1) ** 6))
    term2 = a1_tr_3d * (np.exp(HUBC2 * U) - (HUBC1 ** 3 - np.cbrt(U)))

    # Compute tr_3d
    tr_3d = c0_tr_3d + term1 + term2

    return tr_3d


def calculate_tr_2p(U, HUBC1, HUBC2):
    # Calculate each term of the tr_2p formula
    term1 = a0_tr_2p * (np.cos(np.sqrt(U)) - ((U / HUBC1) * np.sin(HUBC2)))

    # Compute tr_2p
    tr_2p = c0_tr_2p + term1

    return tr_2p

def calculate_svm1(U, HUBC1, HUBC2):
    # Calculate tr_3d and tr_2p for the given U, HUBC1, HUBC2
    tr_3d_normalised = calculate_tr_3d_norm(U, HUBC1, HUBC2)
    tr_2p_normalised = calculate_tr_2p_norm(U, HUBC1, HUBC2)

    # Normalize U
    normalized_U = (U - U_min) / U_range

    # Calculate SVM1 using the normalized values
    svm1_value = (normalized_U * A1) + (tr_3d_normalised * B1) + (tr_2p_normalised * C1) + D1

    return svm1_value

def calculate_svm2(U, HUBC1, HUBC2):
    # Calculate tr_3d and tr_2p for the given U, HUBC1, HUBC2
    tr_3d_normalised = calculate_tr_3d_norm(U, HUBC1, HUBC2)
    tr_2p_normalised = calculate_tr_2p_norm(U, HUBC1, HUBC2)

    # Normalize U
    normalized_U = (U - U_min) / U_range

    # Calculate SVM1 using the normalized values
    svm2_value = (normalized_U * A2) + (tr_3d_normalised * B2) + (tr_2p_normalised * C2) + D2

    return svm2_value

# Function to calculate sum of squares of percentage errors
def sum_of_squares_percentage_errors(U, HUBC1, HUBC2):
    # Calculate DSS, band gap (BG), and v0 using the specified U, HUBC1, HUBC2
    computed_DSS = calculate_DSS(U, HUBC1, HUBC2)
    computed_BG = calculate_band_gap(U, HUBC1, HUBC2)
    computed_v0 = calculate_v0(U, HUBC1, HUBC2)
    computed_tr_3d = calculate_tr_3d(U, HUBC1, HUBC2)
    computed_tr_2p = calculate_tr_2p(U, HUBC1, HUBC2)

    # Print calculated values
    #print(f"Calculated Band Gap (BG): {computed_BG}")
    #print(f"Calculated DSS: {computed_DSS}")
    #print(f"Calculated v0: {computed_v0}")

    # Calculate percentage errors
    percent_error_DSS = 100 * ((computed_DSS - DSS_ref) / DSS_ref)
    percent_error_BG = 100 * ((computed_BG - BG_ref) / BG_ref)
    percent_error_v0 = 100 * ((computed_v0 - v0_ref) / v0_ref)
    percent_error_tr_3d = 100 * ((computed_tr_3d - ref_3d) / ref_3d)
    percent_error_tr_2p = 100 * ((computed_tr_2p - ref_2p) / ref_2p)

    I = percent_error_BG
    J = percent_error_v0
    VM2 = np.sqrt(I**2 + J**2)

    K = 1000/(U+HUBC1)
    #L = percent_error_tr_3d
    #M = percent_error_tr_2p

    # Compute sum of squares of percentage errors
    sum_of_squares = np.sqrt(VM2**2 + K**2)

    return sum_of_squares


bounds = [
    {'name': 'U', 'type': 'continuous', 'domain': (0.5, 5.0)},
    {'name': 'HUBC1', 'type': 'continuous', 'domain': (0, 1.3)},
    {'name': 'HUBC2', 'type': 'continuous', 'domain': (-0.6, 0.0)}
]

def objective_function(x):
    U, HUBC1, HUBC2 = x[0]

    # Check SVM constraints
    svm1_value = calculate_svm1(U, HUBC1, HUBC2)
    svm2_value = calculate_svm2(U, HUBC1, HUBC2)

    if svm1_value < -9.84 or svm2_value < -9.16:
        # Return a large value to penalize violation of constraints
        return 1e10  # Use a large penalty value

    tr_3d_value = calculate_tr_3d(U,HUBC1,HUBC2)
    tr_2p_value = calculate_tr_2p(U, HUBC1, HUBC2)

    if tr_3d_value < 0 or tr_2p_value < 0:
        return 1e10

    # Check if percentage errors exceed 50%
    percent_error_tr_3d = 100 * ((tr_3d_value - ref_3d) / ref_3d)
    percent_error_tr_2p = 100 * ((tr_2p_value - ref_2p) / ref_2p)

    if abs(percent_error_tr_3d) > 50 or abs(percent_error_tr_2p) > 50:
        # Return a large value to penalize violation of percentage error constraints
        return 1e10  # Use a large penalty value

    # Calculate sum of squares of percentage errors
    sum_of_squares = sum_of_squares_percentage_errors(U, HUBC1, HUBC2)

    return sum_of_squares

def run_optimization(max_iter):
    # Define the output file path
    output_file = "optimised_results.txt"

    # Open the output file in append mode
    with open(output_file, 'a') as f:
        for iteration in range(max_iter):

            # Create the GPyOpt optimizer
            optimizer = GPyOpt.methods.BayesianOptimization(
                f=objective_function,
                domain=bounds,
                acquisition_type='EI',  # Expected Improvement
                exact_feval=True
            )

            # Run optimization for a single iteration
            optimizer.run_optimization(max_iter=1, verbosity=False)

            # Get the optimized parameters and function value
            best_params = optimizer.X[np.argmin(optimizer.Y)]
            best_objective_value = np.min(optimizer.Y)

            # Calculate corresponding predicted values
            predicted_band_gap = calculate_band_gap(*best_params)
            predicted_v0 = calculate_v0(*best_params)

            # Check corresponding SISSO predicted 3d and 2p trace
            predicted_3d = calculate_tr_3d(*best_params)
            predicted_2p = calculate_tr_2p(*best_params)

            # Check SVM constraints with the best parameters found
            svm1_value = calculate_svm1(*best_params)
            svm2_value = calculate_svm2(*best_params)

            tr_3d_value = calculate_tr_3d(*best_params)
            tr_2p_value = calculate_tr_2p(*best_params)

            percent_error_tr_3d = 100 * ((tr_3d_value - ref_3d) / ref_3d)
            percent_error_tr_2p = 100 * ((tr_2p_value - ref_2p) / ref_2p)

            # Determine the message based on SVM constraint violations
            svm_message = "Stable"
            if svm1_value < -9.84 or svm2_value < -9.16:
                svm_message = "Unstable"

            om_tr_message = "Nonzero"
            if tr_3d_value < 0 or tr_2p_value < 0:
                om_tr_message = "Zero"

            hybrid_error_message = "Sensible"
            if abs(percent_error_tr_3d) > 50 or abs(percent_error_tr_2p) > 50:
                hybrid_error_message = "Unphysical"

            # Write the iteration's data to the output file
            f.write(f"{iteration + 1} {best_params[0]} {best_params[1]} {best_params[2]} "
                    f"{predicted_band_gap} {predicted_v0} {predicted_3d} {predicted_2p} {svm_message} {om_tr_message} {hybrid_error_message} {best_objective_value}\n")

            # Print status message to terminal
            print(f"Iteration {iteration + 1}/{max_iter} - Results appended to '{output_file}'")

    print("Optimization process completed.")

# Example usage:
max_iterations = 5  # Set the number of iterations for Bayesian optimization
run_optimization(max_iterations)