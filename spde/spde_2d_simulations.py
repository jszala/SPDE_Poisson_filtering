
import numpy as np
from tqdm import tqdm



###########################################################################################################################################################################################################################################################################




##### TO DO:
########### 1. Implement implicit versions
########### 2. Connect with SDE module?

# useful function which returns the indices of the k-th lower diagonal (-k for upper diagonal) of a numpy array.
def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
    


###########################################################################################################################################################################################################################################################################

# forward step functions

# drift
def forward_add_coeff(state, time, coefficient):
    return coefficient*state


# needed for e.g. laplacian
def forward_linear_operator(state, time, linear_operator):
    return np.matmul(linear_operator,state)

# noise
def forward_add_noise(state, time, coefficient):
    return coefficient*np.random.normal(0,1, size=state.shape[0])

# returns correlated noise in space (input SQUARE ROOT of covariance matrix!!)
def forward_add_correlated_noise(state, time, coefficient, sqrt_covariance_matrix):
    return coefficient*np.matmul(sqrt_covariance_matrix, np.random.normal(0,1, size=state.shape[0]))

# simple multiplicative noise
def forward_mul_noise_coefficient(state, time, coefficient):
    return coefficient*state*np.random.normal(0,1, size=state.shape[0])

# returns correlated noise version of above (input SQUARE ROOT of covariance matrix!!)
def forward_mul_correlated_noise_coefficient(state, time, coefficient, sqrt_covariance_matrix):
    return coefficient*state*np.matmul(sqrt_covariance_matrix, np.random.normal(0,1, size=state.shape[0]))



# for FHN equations

def fitzhugh_nagumo_nonlinearity(inhibitor, x_1, x_2, x_3):
    return np.multiply( np.multiply(inhibitor - x_1, x_3- inhibitor), inhibitor - x_2 )

# fhn inhibitor
def forward_fhn_activator(state, time, laplacian_inh, alpha, skew, potential):
    return np.matmul(laplacian_inh,state[0]) + fitzhugh_nagumo_nonlinearity(state[0], *alpha)*skew - state[1] + potential


# fhn activator
def forward_fhn_inhibitor(state, time, laplacian_act, beta, gamma):
    return np.matmul(laplacian_act,state[1]) + gamma*(beta*state[0] - state[1])








###########################################################################################################################################################################################################################################################################

# creates discrete 2d Neumann Laplacian
# Default is the symmetric version
def calc_neumann_laplacian_2d(number_pixels_vertical, number_pixels_horizontal, diffusion, ghost_nodes = False):

    # symmetric version
    if ghost_nodes == False:
    
        laplacian = np.zeros((number_pixels_vertical*number_pixels_horizontal, number_pixels_vertical*number_pixels_horizontal))

        # main diagonal
        laplacian[kth_diag_indices(laplacian, 0)] = np.hstack([np.hstack([2,np.tile(3,number_pixels_horizontal-2),2]), np.tile(np.hstack([3,np.tile(4,number_pixels_horizontal-2),3]), number_pixels_vertical-2), 
                                                        np.hstack([2,np.tile(3,number_pixels_horizontal-2),2]) ])

        # left and right pixels
        laplacian[kth_diag_indices(laplacian, 1)] =  np.hstack([np.tile(np.hstack([np.tile(-1,number_pixels_horizontal-1),0]), number_pixels_vertical-1), np.hstack([np.tile(-1,number_pixels_horizontal-1)]) ])
        laplacian[kth_diag_indices(laplacian, -1)] = np.hstack([np.tile(np.hstack([np.tile(-1,number_pixels_horizontal-1),0]), number_pixels_vertical-1), np.hstack([np.tile(-1,number_pixels_horizontal-1)]) ])
        
        # upper and lower pixels
        laplacian[kth_diag_indices(laplacian, number_pixels_horizontal)] = -1
        laplacian[kth_diag_indices(laplacian, -number_pixels_horizontal)] = -1


    # version using ghost nodes at the border
    elif ghost_nodes == True:
        # main diagonal
        laplacian = 4*np.eye(number_pixels_vertical*number_pixels_horizontal)

        # left and right pixels
        laplacian[kth_diag_indices(laplacian, 1)] = np.hstack([np.tile(np.hstack([-2,np.tile(-1,number_pixels_horizontal-2),0]), number_pixels_vertical-1), np.hstack([-2,np.tile(-1,number_pixels_horizontal-2)]) ])
        laplacian[kth_diag_indices(laplacian, -1)] = np.hstack([np.hstack([np.tile(-1,number_pixels_horizontal-2),-2]) , np.tile(np.hstack([0,np.tile(-1,number_pixels_horizontal-2),-2]), number_pixels_vertical-1)])

        # upper and lower pixels
        laplacian[kth_diag_indices(laplacian, number_pixels_horizontal)] =    np.hstack([np.tile(-2,number_pixels_horizontal), np.tile(np.tile(-1,number_pixels_horizontal), number_pixels_vertical-2)]) 
        laplacian[kth_diag_indices(laplacian, -number_pixels_horizontal)] =    np.hstack([np.tile(np.tile(-1,number_pixels_horizontal), number_pixels_vertical-2), np.tile(-2,number_pixels_horizontal)]) 



    return diffusion*laplacian




# calculates a simple "Gauss kernel" covariance matrix, see notebook Smooth_noise_building
# matrix is positive semidefinite
def return_simple_covariance_matrix(covariance_diameter, decaying_factor, dim_x, dim_y, return_sqrt = True):

    roll_matrix = np.zeros([dim_x,dim_y])
    shift_factor = int((covariance_diameter-1)/2)
    center_pixel = shift_factor 


    # this matrix simulates a "Gaussglocke" around one pixel
    roll_matrix[center_pixel, center_pixel] = 1

    for k in np.arange(0,2*shift_factor+1):
        for l in  np.arange(0,2*shift_factor+1):
            roll_matrix[k,l] = np.exp(-decaying_factor*(np.sqrt((center_pixel-k)**2 +(center_pixel-l)**2)))



    # this correlation Glocke will be rolled through every pixel
    roll_array = roll_matrix.flatten()

    
    total_dim = dim_x*dim_y
    
    #actual covariance matrix
    cov_matrix = np.zeros([total_dim, total_dim])

    for i in np.arange(total_dim):
        cov_matrix[i] = np.roll(roll_array, i )

    # needs to be shifted to center main diag around center pixels
    shift_factor = int((covariance_diameter-1)/2)
    #shift_factor = 1

    cov_matrix = np.roll(cov_matrix, -(shift_factor*dim_x + shift_factor), axis = 1)

    # delete covariances of edges across the image
    cov_matrix[:-(total_dim-(shift_factor*dim_x + shift_factor)), total_dim-(shift_factor*dim_x + shift_factor):] = 0
    cov_matrix[total_dim-(shift_factor*dim_x + shift_factor):, :-(total_dim-(shift_factor*dim_x + shift_factor))] = 0 
    
    if return_sqrt == True:
        # returns sqrt of covariance matrix e.g. for Euler scheme
        return np.sqrt(cov_matrix)
    else:
        return cov_matrix
    
    
    

###########################################################################################################################################################################################################################################################################

# Types of forward steps

# used for scalar signal such as heat equations
def forward_step_standard(state, time_delta, time, functions, functions_args, implicit = False):
    if implicit == False:
        # functions[0]: drift coefficient
        # functions[1]: noise_coefficient
        return state + time_delta*functions[0](state, time, *functions_args[0]) + np.sqrt(time_delta)*functions[1](state, time, *functions_args[1])




# for Fitzhugh-Nagumo

# functions[0]: drift coefficient activator
# functions[1]: noise_coefficient activator

# functions[2]: drift coefficient inhibitor
# functions[3]: noise_coefficient inhibitor

def forward_step_fhn(state, time_delta, time, functions, functions_args, implicit = False):
    if implicit == False:
        activator = state[0] + time_delta*functions[0](state, time, *functions_args[0]) + np.sqrt(time_delta)*functions[1](state[0], time, *functions_args[1])
        inhibitor = state[1] + time_delta*functions[2](state, time, *functions_args[2]) + np.sqrt(time_delta)*functions[3](state[1], time, *functions_args[3])

    # returns both inhibitor and activator
    return [activator, inhibitor]






###########################################################################################################################################################################################################################################################################


# main function to simulate an SPDE path (here: signal) 

### could also be used to simulate SDE

def simulate_signal_path(time_delta: float, time_steps: int, dimension : int,  initial_condition,  forward_step, coefficient_functions, coefficient_function_arguments, type = "standard", implicit = False, time_homog = True, print_steps = False):
    
    # for simple dynamics such as heat equations
    if type == "standard":
        signal_path = np.zeros([time_steps, dimension])
        signal_path[0] = initial_condition

    # for activator inhibitor models such as Fitzhugh-Nagumo
    if type == "act_inh":
        
        activator_path = np.zeros([time_steps, dimension])
        activator_path[0] = initial_condition[0]

        inhibitor_path = np.zeros([time_steps, dimension])
        inhibitor_path[0] = initial_condition[1]


    if implicit == False:
        if time_homog == True:
            for i in tqdm(np.arange(1,time_steps), disable = not print_steps):
                if type == "standard":
                    signal_path[i] =  forward_step(signal_path[i-1], time_delta, i*time_delta, coefficient_functions, coefficient_function_arguments)

                if type == "act_inh":
                    activator_path[i], inhibitor_path[i] =  forward_step([activator_path[i-1], inhibitor_path[i-1]], time_delta, i*time_delta, coefficient_functions, coefficient_function_arguments)
    
    if type == "standard":
        return signal_path
    
    if type == "act_inh":
        return [activator_path, inhibitor_path]