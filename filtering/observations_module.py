import numpy as np


###########################################################################################################################################################################################################################################################################


# this matrix can be used to generate a "pixelated" observation. scaling factor k -> reduce pixel number by pixels/k**2
# the matrix will go over disjoint(!) pixel areas whose size is determined by scaling_factor

def return_downscaling_matrix(scaling_factor: int, old_dim_x: int, old_dim_y: int, type ="mean", dx_old = 1,  partial_method="center"):
    
    new_dim_x = int(old_dim_x/scaling_factor)
    new_dim_y = int(old_dim_y/scaling_factor)
    
    old_dim_flattened = old_dim_x*old_dim_y
    new_dim_flattened = new_dim_x*new_dim_y


    downscaling_matrix = np.zeros([ new_dim_flattened, old_dim_flattened])
    
    if type == "mean":
        normalization_factor = 1/scaling_factor**2

        k = 0
        for i in np.arange(new_dim_x):
            for j in np.arange(new_dim_y):
                roll_matrix = np.zeros([old_dim_x, old_dim_y])
                roll_matrix[scaling_factor*i:scaling_factor*(i+1),scaling_factor*j:scaling_factor*(j+1)] = normalization_factor
                downscaling_matrix[k] = roll_matrix.flatten()
                k+=1

    # this type will sum up the pixels and return one summed up pixel per 'scaling factor area'
    elif type == "sum":

        k = 0
        for i in np.arange(new_dim_x):
            for j in np.arange(new_dim_y):
                roll_matrix = np.zeros([old_dim_x, old_dim_y])
                roll_matrix[scaling_factor*i:scaling_factor*(i+1),scaling_factor*j:scaling_factor*(j+1)] = 1
                downscaling_matrix[k] = roll_matrix.flatten()
                k+=1
    
    # this type will integrate over the "scaling factor area". The physical length of a pixel has to be passed by "dx_old"
    elif type == "integration":
        
        integration_constant = dx_old**2

        k = 0
        for i in np.arange(new_dim_x):
            for j in np.arange(new_dim_y):
                roll_matrix = np.zeros([old_dim_x, old_dim_y])
                roll_matrix[scaling_factor*i:scaling_factor*(i+1),scaling_factor*j:scaling_factor*(j+1)] = integration_constant
                downscaling_matrix[k] = roll_matrix.flatten()
                k+=1

    elif type == "partial":
        
        integration_constant = dx_old**2

        k = 0
        for i in np.arange(new_dim_x):
            for j in np.arange(new_dim_y):
                roll_matrix = np.zeros([old_dim_x, old_dim_y])
                if partial_method == "corner":
                    roll_matrix[scaling_factor*i,scaling_factor*j] = integration_constant
                elif partial_method == "center":
                    roll_matrix[int(np.floor(0.5*(scaling_factor*i+scaling_factor*(i+1)))), int(np.floor(0.5*(scaling_factor*j+scaling_factor*(j+1))))] = integration_constant
                downscaling_matrix[k] = roll_matrix.flatten()
                k+=1

    else:
        print("Downscaling type not implemented")
    
    return downscaling_matrix   



###########################################################################################################################################################################################################################################################################

# this is an ensemble version of matrix multiplication
### NEED TO CHANGE


def mul_multiply_with_matrix(state, time,  matrix, ensemble_size = 1):
    
        if ensemble_size == 1:
            return np.matmul(matrix, state)
        else:
            return_array =  np.matmul(matrix, state[0])
            
            for i in np.arange(1,ensemble_size):
                return_array = np.vstack([return_array, np.matmul(matrix, state[i])])

            return return_array