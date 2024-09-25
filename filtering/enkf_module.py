import numpy as np
import scipy
from tqdm import tqdm



# useful function which returns the indices of the k-th lower diagonal (-k for upper diagonal) of a numpy array.
def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols
# for matrix free version of  EnKF



###########################################################################################################################################################################################################################################################################


# A_0 has to be diagonal 
def sherman_morrison_solver(A_0, U, V, b):
    # observation size
    m = A_0.shape[0]
    
    # ensemble size
    N = U.shape[1] 


    # solution
    x = np.zeros([N+1,m])
    
    # auxiliary matrix
    y = np.zeros([N+1, N+1, m])

    A_0_inv = np.zeros_like(A_0)
    A_0_inv[kth_diag_indices(A_0_inv,0)] = np.divide(1, np.diag(A_0))

    x[0] = np.matmul(A_0_inv, b)
    

    for k in np.arange(1,N+1):
        y[0,k] =  np.matmul(A_0_inv, U[:,k-1])
        

    for i in np.arange(1,N):

        x[i] = x[i-1] - np.divide(np.matmul(V[:,i-1], x[i-1])*y[i-1,i], 1+np.matmul(V[:,i-1], y[i-1,i]))
        


        for l in np.arange(i+1, N+1):
           
            y[i,l] = y[i-1,l] -  np.multiply(np.divide(np.matmul(V[:,i-1], y[i-1,l]), 1+np.matmul(V[:,i-1], y[i-1,i])), y[i-1,i])

       
    
    
    x[N] = x[N-1] - np.multiply(np.divide(np.matmul(V[:,N-1], x[N-1]), 1+np.matmul(V[:,N-1], y[N-1,N])), y[N-1,N])
    
    y_store = np.zeros([N,m])
    
    for i in np.arange(1,N+1):
       y_store[i-1] = y[i-1, i]

    return x[-1], y_store


def simplified_sherman_morrison_solver(A_0, V, b, y):
    # observation size

    m = A_0.shape[0]
    
    # ensemble size
    N = V.shape[1]

    # solution
    x = np.zeros([N+1,m])

    A_0_inv = np.zeros_like(A_0)
    A_0_inv[kth_diag_indices(A_0_inv,0)] = np.divide(1, np.diag(A_0))

    x[0] = np.matmul(A_0_inv, b)    

    for i in np.arange(1,N+1):
        x[i] = x[i-1] - np.divide(np.matmul(V[:,i-1], x[i-1])*y[i-1], 1+np.matmul(V[:,i-1], y[i-1]))


    return x[-1]




###########################################################################################################################################################################################################################################################################


def enkf_initialize(signal_size, ensemble_size, initial_mean, initial_cov):

    ensemble_initital = np.tile(initial_mean, [ensemble_size,1])

    for i in np.arange(ensemble_size):
        ensemble_initital[i] = initial_mean + np.matmul(np.sqrt(initial_cov),np.random.normal(0, scale = 1, size = (signal_size)))

    return ensemble_initital


###########################################################################################################################################################################################################################################################################




def enkf_estimate(observations, ensemble_size, time_delta, time_skip_step, signal_type, signal_args, 
    observation_drift_function, observation_drift_function_arguments, observation_noise_coeff, static_coefficients = True, print_steps = True, implicit = True, matrix_free = False, poisson_signal = True):    

    time_steps = observations.shape[0]
    observation_size = observations.shape[1]


    
    if signal_type == "standard":
        signal_size = signal_args[0].shape[0]
    elif signal_type == "act_inh":
        signal_size = signal_args[0][0].shape[0]


    # intitalize ensemble
    # to save space we use two time steps in signal ensemble which are filled alternating using time_index modulo 2
    signal_ensemble = np.zeros((2, ensemble_size, signal_size))

    
    # this will be used in the forward step of the signal in case we need to do multiple Euler forward steps per observation time step due to numerical reasons
    time_skip_array = np.zeros((2, signal_size))
    time_delta_signal = time_delta/time_skip_step

    if signal_type == "act_inh":
        time_skip_activator = np.zeros((2, signal_size))



    if signal_type == "standard":
        signal_ensemble[0] = enkf_initialize( signal_size, ensemble_size, signal_args[0], signal_args[1])
    elif signal_type == "act_inh":
        signal_ensemble[0] = enkf_initialize( signal_size, ensemble_size, signal_args[0][0], signal_args[1][0])

    # intitalize forward process
    signal_forward = np.zeros((ensemble_size, signal_size))

    # intitialize filtered process
    mean_estimate = np.zeros([time_steps, signal_size])

    if poisson_signal == True:
        integrated_process  = np.zeros((ensemble_size, observation_size))
        

    forward_step_signal = signal_args[2]
    signal_coefficient_functions = signal_args[3]
    signal_coefficient_function_arguments = signal_args[4]
        


    # need an additional process for Fitzhugh Nagumo dynamics
    if signal_type == "act_inh":
        activator_ensemble = np.zeros_like(signal_ensemble)
        activator_ensemble[0] = enkf_initialize( signal_size, ensemble_size, signal_args[0][1], signal_args[1][1])
    
    if static_coefficients == True:

        observation_covariance = time_delta*np.matmul(observation_noise_coeff, observation_noise_coeff.T)


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
                            time_skip_array[skip%2], time_skip_activator[skip%2] = forward_step_signal(np.array([time_skip_array[(skip-1)%2], time_skip_activator[(skip-1)%2]]), time_delta_signal, i*time_delta + skip*time_delta_signal, signal_coefficient_functions, signal_coefficient_function_arguments)
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

            # calculate measurement matrix
            measurement = np.tile(observations[i] - observations[i-1], (ensemble_size,1)) + np.sqrt(time_delta)*np.matmul(observation_noise_coeff, np.random.normal(0,1,(observation_size, ensemble_size))).T
           
 
            mean_forward = signal_forward.mean(axis = 0).reshape((1,signal_size))
            
            # calculate ensemble covariance
            deviation = signal_forward - mean_forward
 
            
            # calculate the analyzed particles       
            
            if poisson_signal == True:
                # for h_Ito function
                # observation_drift_function_arguments[0] is the intensity function, e.g. lambda(x) = (c*x)^2
                # observation_drift_function_arguments[1] are the intensity function arguments
                HX = time_delta*observation_drift_function(signal_forward.copy(), i*time_delta, observation_drift_function_arguments[0], observation_drift_function_arguments[1], integrated_process, ensemble_size)
            else:
                HX = time_delta*observation_drift_function(signal_forward.copy(), i*time_delta,*observation_drift_function_arguments,  ensemble_size)
            

            HA =  HX - HX.mean(axis = 0).reshape((1,observation_size))
            error_term = measurement - HX
            
            
            

            
            
            if matrix_free == False:
                P = observation_covariance + np.divide(1, ensemble_size-1)*np.matmul(HA.T, HA)
                M = scipy.linalg.solve(P, error_term.T)             
                Z = np.matmul(HA, M)
                W = np.divide(1, ensemble_size -1) * np.matmul(deviation.T, Z).T
                signal_ensemble[i%2] = signal_forward + W

            
            else:
                #

                # Sherman Morrison solver for high dimensional data
                # first iteration
                Z_aux, y = sherman_morrison_solver(observation_covariance, HA.T, HA.T, error_term[0])

                mat_aux = np.matmul(deviation.T, HA)
                W = np.matmul(mat_aux, Z_aux)
                
                # analyzed particle
                signal_ensemble[i%2, 0] = signal_forward[0] + W


                # other iterations    
                for k in np.arange(1,ensemble_size):
                    Z_aux = simplified_sherman_morrison_solver(observation_covariance, HA.T, error_term[k], y)
                    W = np.matmul(mat_aux, Z_aux)
                    # analyzed particles
                    signal_ensemble[i%2, k] = signal_forward[k] + W
                

            # estimate
            mean_estimate[i] = signal_ensemble[i%2].mean(axis = 0).reshape((signal_size))
            
            if poisson_signal == True:
                # this is an approximation of the compensator of the jump process by the process particles themselves
                for ensemble_index in np.arange(ensemble_size):
                    integrated_process[ensemble_index] += time_delta*observation_drift_function_arguments[0](signal_ensemble[i%2, ensemble_index], i*time_delta, *observation_drift_function_arguments[1])
            
          
    else:
        print("Not implemented.")
        return

    
    return mean_estimate