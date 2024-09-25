import numpy as np
from tqdm import tqdm





# function to compute the initial condition of the signal in filter
#
# signal_size: int, ensemble_size: int, initial_mean: np.array(signal_size), initial_cov: np.array([signal_size, signal_size])
#
#
def signal_initialize(signal_size, ensemble_size, initial_mean, initial_cov):

    ensemble_initital = np.tile(initial_mean, [ensemble_size,1])

    for i in np.arange(ensemble_size):
        ensemble_initital[i] = initial_mean + np.matmul(np.sqrt(initial_cov),np.random.normal(0, scale = 1, size = (signal_size)))

    return ensemble_initital


## analysis step for particle filter

# resampling step copied from the package filterpy due to the package not loading correctly
def residual_resample(weights):
    """ Performs the residual resampling algorithm used by particle filters.

    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]


    Parameters
    ----------

    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.

    References
    ----------

    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))

    return indexes




# computes the exponent of the Radon-Nikodym density for a point process w.r.t. a Poisson process with rate = pixel_area_size

# input : forward_particle np.array(observation_size), jump_increment np.array()
def poisson_weight_function(forward_particle, jump_increment, pixel_area_size, dt):

    poisson_integral = 0
    
    for i in np.arange(jump_increment.shape[0]):

        poisson_integral += jump_increment[i]*np.log(forward_particle[i]/pixel_area_size) - dt*(forward_particle[i]-pixel_area_size)

    
    return poisson_integral
    

# computes log{ sum_j exp{exponent[j]} } in a numerically feasible way (log-sum-exp trick)
def calc_log_nominator(exponents):

    exponent_max = exponents.max()

    log_sum = 0

    for i in np.arange(exponents.shape[0]):
        log_sum += np.exp(exponents[i]-exponent_max)
    
    return exponent_max + np.log(log_sum)


# This function computes the weights of the particles in the particle filter algorithm
# The weights have to have an exponential form, i.e. weight[i] = exp{weight_function[i]}/(sum_j exp{weight_function[j]})
# To avoid overflow, we use that log(weight[i]) = weight_function[i] - log{ sum_j exp{weight_function[j]} } and the  log-sum-exp trick.

# input: forward_particles np.array(ensemble_size, observation_size), observation_increment np.array(observation_size), weight_function : either gauss_weight_function or poisson_weight_function , weight_function_arguments: list of corresponding arguments
# output: weights np.array(ensemble_size)

def compute_weights(forward_particles, observation_increment, weight_function, weight_function_arguments):
    
    ensemble_size = forward_particles.shape[0]

    exponents = np.empty(ensemble_size)  
    log_weights = np.empty(ensemble_size)
    weights = np.empty(ensemble_size)

    # compute the exponents 
    for particle_number in np.arange(ensemble_size):
        exponents[particle_number] = weight_function(forward_particles[particle_number,:], observation_increment, *weight_function_arguments)

    # compute the log of the nominator log{ sum_j exp{weight_function[j]} }
    log_sum = calc_log_nominator(exponents)

    #  log(weight[i]) = weight_function[i] - log{ sum_j exp{weight_function[j]} }
    for particle_number in np.arange(ensemble_size):
        log_weights[particle_number] = exponents[particle_number] - log_sum

    # the power function yields the actual weights
    weights = np.exp(log_weights)

    
    return weights

#####################################################################################################################################################################################################################################################################################################################################


# particle filter algorithm

def particle_filter_estimate(observations, ensemble_size, time_delta, time_skip_step, signal_type, signal_args, 
    observation_function, observation_function_arguments, weight_function, weight_function_args, static_coefficients = True, print_steps = True, implicit = True,):    

    time_steps = observations.shape[0]
    observation_size = observations.shape[1]


    
    if signal_type == "standard":
        signal_size = signal_args[0].shape[0]
    elif signal_type == "act_inh":
        signal_size = signal_args[0][0].shape[0]


    # intitalize ensemble
    # to save memory we use two time steps in signal ensemble which are filled alternating using time_index modulo 2
    signal_ensemble = np.zeros((2, ensemble_size, signal_size))

    
    # this will be used in the forward step of the signal in case we need to do multiple Euler forward steps per observation time step due to numerical reasons
    time_skip_array = np.zeros((2, signal_size))
    time_delta_signal = time_delta/time_skip_step

    if signal_type == "act_inh":
        time_skip_activator = np.zeros((2, signal_size))



    if signal_type == "standard":
        signal_ensemble[0] = signal_initialize( signal_size, ensemble_size, signal_args[0], signal_args[1])
    elif signal_type == "act_inh":
        signal_ensemble[0] = signal_initialize( signal_size, ensemble_size, signal_args[0][0], signal_args[1][0])

    # intitalize forward process
    signal_forward = np.zeros((ensemble_size, signal_size))

    # intitialize filtered process
    mean_estimate = np.zeros([time_steps, signal_size])


        

    forward_step_signal = signal_args[2]
    signal_coefficient_functions = signal_args[3]
    signal_coefficient_function_arguments = signal_args[4]
        


    # need an additional process for Fitzhugh Nagumo dynamics
    if signal_type == "act_inh":
        activator_ensemble = np.zeros_like(signal_ensemble)
        activator_ensemble[0] = signal_initialize( signal_size, ensemble_size, signal_args[0][1], signal_args[1][1])
    

    



    measurement_ensemble = np.zeros((ensemble_size, observation_size))
    weights = np.ones([2, ensemble_size])*1/ensemble_size


    if static_coefficients == True:

        


        if signal_type == "standard":
            mean_estimate[0] = signal_args[0]
        elif signal_type == "act_inh":
            mean_estimate[0] = signal_args[0][0]

   

        for i in tqdm(np.arange(1,time_steps), disable = not print_steps):
            # forward step
            for k in np.arange(ensemble_size):
                if implicit == False:
                    
                    skip_last = 0
                    
                    # if time_skip_step == 1 this is just a normal Euler forward step
                    if signal_type =="standard":
                        time_skip_array[skip_last] = forward_step_signal(signal_ensemble[(i-1)%2,k] , time_delta_signal, i*time_delta, signal_coefficient_functions, signal_coefficient_function_arguments)
                    else:
                        time_skip_array[skip_last], time_skip_activator[skip_last] = forward_step_signal(np.array([signal_ensemble[(i-1)%2,k], activator_ensemble[(i-1)%2,k]]) , time_delta_signal, i*time_delta, signal_coefficient_functions, signal_coefficient_function_arguments)

                    

                    for skip in np.arange(1,time_skip_step):
                    # we do 'time_skip_step' Euler steps due to numerical reasons if needed
                        if signal_type == "standard":
                            time_skip_array[skip%2] = forward_step_signal(time_skip_array[(skip-1)%2], time_delta_signal, i*time_delta+ skip*time_delta_signal, signal_coefficient_functions, signal_coefficient_function_arguments)
                            skip_last = skip
                        else:
                            time_skip_array[skip%2], time_skip_activator[skip%2] = forward_step_signal(np.array([time_skip_array[(skip-1)%2].copy(), time_skip_activator[(skip-1)%2].copy()]), time_delta_signal, i*time_delta + skip*time_delta_signal, signal_coefficient_functions, signal_coefficient_function_arguments)
                            skip_last = skip

   

                    if signal_type == "standard":
                        signal_forward[k] = time_skip_array[skip_last%2]
                    else:
                        signal_forward[k]=  time_skip_array[skip_last%2]
                        activator_ensemble[i%2,k] = time_skip_activator[skip_last%2]


                else:
                    print("Not yet implemented")
                    return

                     


            # analysis step

            #calculate measurements

            for ensemble_member in np.arange(ensemble_size):

                measurement_ensemble[ensemble_member] = observation_function(signal_forward[ensemble_member], i*time_delta, *observation_function_arguments)

            #calculate weights
            for ensemble_member in np.arange(ensemble_size):
                weights[i%2] = compute_weights(measurement_ensemble, observations[i] - observations[i-1], weight_function, weight_function_args)


    
            # resampling step
            resampled_indices = residual_resample(weights[i%2])


            # only particles acc. to the resampling are kept 
            signal_ensemble[i%2] = signal_forward[resampled_indices]
            


            # compute mean of particles for estimate
            mean_estimate[i] = signal_ensemble[i%2].mean(axis = 0).reshape((signal_size))


    
    return mean_estimate