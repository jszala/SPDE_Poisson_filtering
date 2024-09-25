import numpy as np


########################################################################################################################################################

# Intensity modules

# time_homogeneous

def hom_intensity_power(signal_state, time, downscaling_matrix, coefficient, exponent):
    return np.matmul(downscaling_matrix,np.power(coefficient*signal_state, exponent))

def hom_intensity_coefficient(signal_state, time, downscaling_matrix, coefficient):
    return np.matmul(downscaling_matrix, coefficient*signal_state)


# time inhomogeneous 

# exponential decay versions of above
def inhom_intensity_exponential_coefficient(signal_state, time, downscaling_matrix, coefficient, exponential_coefficient):
    return  np.matmul(downscaling_matrix, np.exp(exponential_coefficient*time)*coefficient*signal_state)

def inhom_intensity_exponential_power(signal_state, time, downscaling_matrix, coefficient, power_exponent, exponential_coefficient):
    return  np.matmul(downscaling_matrix, np.exp(exponential_coefficient*time)*np.power(coefficient*signal_state, power_exponent))


# calculates whole intensity path
# in each time step we apply intensity_function with intensity_function_args (list of arguments)

def calculate_intensity_path(signal_path, dim_intensity, Time_steps, time_delta,  intensity_function, intensity_function_args):
    intensity_path = np.zeros([Time_steps, dim_intensity])

    for i in np.arange(Time_steps):
        intensity_path[i]  = intensity_function(signal_path[i], i*time_delta, *intensity_function_args)
    
    return intensity_path


########################################################################################################################################################


# Poisson processes and approximations (see notebook)

''' deprecated
def calculate_cox_process_intensity_from_signal(signal_path, intensity_function, intensity_function_arguments, time_line = 0):
        return intensity_function(signal_path, time_line, *intensity_function_arguments)
'''

# generates whole intensity path
# shift_for_filter == True approximates the intensity for filtering from the rhs s.t. Y_ti \sim Poiss( lambda(X_ti) dt)
# shift_for_filter == False approximates the intensity for from the lhs s.t. Y_ti \sim Poiss( lambda(X_t(i-1)) dt)              
def simulate_cox_process_from_intensity(intensity, time_delta, initial_condition = 0, shift_for_filter = True):
        cox_process_path = np.zeros_like(intensity)
        cox_process_path[0,:] = initial_condition
        
        if shift_for_filter == True:
                for i in np.arange(1,intensity.shape[0]):
                        cox_process_path[i,:] = cox_process_path[i-1,:] + np.random.poisson(time_delta*intensity[i,:])

        else:
                for i in np.arange(1,intensity.shape[0]):
                        cox_process_path[i,:] = cox_process_path[i-1,:] + np.random.poisson(time_delta*intensity[i-1,:])

        return cox_process_path




def calculate_integrated_intensity(intensity, time_delta):
        integrated_intensity = np.zeros_like(intensity)
        integrated_intensity[0,:] = 0

        for time in np.arange(1, intensity.shape[0]):
                integrated_intensity[time,:] = integrated_intensity[time-1, :] + intensity[time,:]*time_delta

        return integrated_intensity



# stabilizing transform
def calculate_stabilizing_transform_of_poisson_process(poisson_process, time_line):
        stabilized_path = np.zeros_like(poisson_process)
        stabilized_path[0,:] = poisson_process[0,:]

        poisson_increments = poisson_process[1:,:] - poisson_process[:-1,:]

        timesteps = poisson_process.shape[0]
        dimension = poisson_process.shape[1]

        last_jump = np.zeros(dimension).astype(int)

        # time loop
        for time in np.arange(1, timesteps):
                # loop over dimension
                for dim in np.arange(dimension):
                        if poisson_increments[time-1,dim] != 0:
                                # jump size is g(t_n, Y_(t_n)) - g(t_(n-1), Y_(t_(n-1))), thus we need knowledge of last jump time
                                stabilized_path[time,dim] = stabilized_path[time-1,dim] + 2*np.sqrt(time_line[time]*poisson_process[time,dim]) - 2*np.sqrt(time_line[last_jump[dim]]*poisson_process[last_jump[dim],dim] ) 
                                last_jump[dim] = time
                        else:
                                stabilized_path[time,dim] = stabilized_path[time-1,dim] 


        return stabilized_path

# Compensator/Ito incements of stabilizing transform

def calculate_compensator_of_stabilizing_transform(poisson_process, intensity, time_line, time_delta):
        compensator_sqrt = np.zeros_like(poisson_process)
        compensator_sqrt[0,:] = 0

        for time in np.arange(1, poisson_process.shape[0]):
                if time > 1:
                        increment = 2*intensity[time-1]*np.sqrt(time_line[time-1])*(np.sqrt(poisson_process[time-1,:] +1) - np.sqrt(poisson_process[time-1,:])) + np.sqrt((1/time_line[time-1])*poisson_process[time-1,:])
                else:
                        increment = 2*intensity[time-1]*np.sqrt(time_line[time-1])*(np.sqrt(poisson_process[time-1,:] +1) - np.sqrt(poisson_process[time-1,:])) 
                
                compensator_sqrt[time,:] = compensator_sqrt[time-1,:] + time_delta*increment

        return compensator_sqrt



def calculate_compensator_approximation(intensity, integrated_intensity, time_line, time_delta):
        compensator_approx = np.zeros_like(intensity)
        compensator_approx[0,:] = 0

        for time in np.arange(1, intensity.shape[0]):
                if time > 1:
                        increment = 2*intensity[time-1]*np.sqrt(time_line[time-1])*(np.sqrt(integrated_intensity[time-1,:] +1) - np.sqrt(integrated_intensity[time-1,:])) + np.sqrt((1/time_line[time-1])*integrated_intensity[time-1,:])
                else:
                        increment = 2*intensity[time-1]*np.sqrt(time_line[time-1])*(np.sqrt(integrated_intensity[time-1,:] +1) - np.sqrt(integrated_intensity[time-1,:])) 
                
                compensator_approx[time,:] = compensator_approx[time-1,:] + time_delta*increment

        return compensator_approx

def calculate_compensator_approximation_kappa(integrated_intensity, time_line):
        
        compensator_approx_kappa = np.zeros_like(integrated_intensity)

        time_steps = integrated_intensity.shape[0]
        dimension = integrated_intensity.shape[1]

        for time in np.arange(1,time_steps):

                if np.min(4*integrated_intensity[time,:]*time_line[time] - time_line[time]) >= 0 :
                        compensator_approx_kappa[time,:] = np.sqrt(4*integrated_intensity[time,:]*time_line[time]- time_line[time])
                else:
                        for dim in np.arange(dimension):
                                if 4*integrated_intensity[time,dim]*time_line[time] - time_line[time] >= 0:
                                        compensator_approx_kappa[time,dim] = np.sqrt(4*integrated_intensity[time,dim]*time_line[time]- time_line[time])
                                else:
                                        compensator_approx_kappa[time,dim] = 0

        return compensator_approx_kappa

def calculate_compensator_approx_chain_rule(intensity, integrated_intensity, time_line, time_delta):

        compensator_approx_cr = np.zeros_like(integrated_intensity)

        time_steps = integrated_intensity.shape[0]

        for time in np.arange(2,time_steps):
                quotient = np.sqrt(integrated_intensity[time-1,:]/time_line[time-1])
                
                increment = intensity[time-1,:]*(1/quotient) + quotient

                compensator_approx_cr[time] = compensator_approx_cr[time-1] + time_delta*increment 

        return compensator_approx_cr     

########################################################################################################################################################


# for filtering with square root transform


def observation_function_poisson(state, time, intensity_function, intensity_function_args, integrated_intensity, ensemble_size):
    

    quotient = np.sqrt(integrated_intensity/time)
    

    # prevents dividing through zero
    if np.any(quotient == 0):
        quotient_inverse = np.zeros_like(quotient)
        if np.any(quotient != 0):
            quotient_inverse[quotient != 0] = np.divide(1,quotient)
            quotient_inverse[quotient == 0] = np.divide(1,quotient+1)
    else:
        quotient_inverse = np.divide(1,quotient)

    if ensemble_size == 1:
        return intensity_function(state, time, *intensity_function_args)*quotient_inverse + quotient
     
    else:
        return_array =  intensity_function(state[0], time, *intensity_function_args)*quotient_inverse[0] + quotient[0]
        for i in np.arange(1,ensemble_size):
            return_array = np.vstack([return_array, intensity_function(state[i], time, *intensity_function_args)*quotient_inverse[i] + quotient[i]])


    return return_array
    