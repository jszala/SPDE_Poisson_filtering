import numpy as np
import pandas as pd
import os


import utils.spde.spde_2d_simulations as spde_2d_simulations

###############################################################################################################################################################################

def save_spde_metadata(working_dir, dt, timesteps, dx, dim_x, dim_y, euler_step_type):
    pd.DataFrame({'dt': dt,
                   'dx':[dx],
                   'timesteps': [timesteps],
                   'dim_x': [dim_x],
                   'dim_y': [dim_y],
                   'Euler_step': euler_step_type}).to_csv(os.path.join(working_dir, 'signal', 'metaparameters.csv'), index = False)

###############################################################################################################################################################################

def load_spde_metadata(working_dir):
    
    metaparameters_df = pd.read_csv(os.path.join(working_dir, 'signal', 'metaparameters.csv'))
    
    #returns in following order dt, dx, timesteps, dim_x, dim_y, eulerstep_type
    return metaparameters_df.values[0,:]

###############################################################################################################################################################################

def save_spde_parameters(working_dir, spde_type, diffusion_coefficient, noise_coefficient, noise_type, covariance_radius = 0, covariance_decay = 0, 
                         diffusion_coefficient_inh = 0 ,  alpha = 0, beta = 0, gamma = 0, skew = 0, potential = 0, noise_coefficient_inh = 0, noise_type_inh = 0, covariance_radius_inh = 0, covariance_decay_inh = 0):


    if spde_type == "fitzhugh_nagumo":
        parameter_df = pd.DataFrame({'spde_type': [spde_type],
                    'diffusion_act': diffusion_coefficient,
                    'noise_type_act': noise_type,
                    'noise_coefficient_act': noise_coefficient,
                    'covariance_radius_act': covariance_radius,
                    'covariance_decay_act': covariance_decay,
                    'diffusion_inh': diffusion_coefficient_inh,
                    'noise_type_inh': noise_type_inh,
                    'noise_coefficient_inh': noise_coefficient_inh,
                    'covariance_radius_inh': covariance_radius_inh,
                    'covariance_decay_inh': covariance_decay_inh,                                            
                    'alpha_1': alpha[0],
                    'alpha_2': alpha[1],
                    'alpha_3': alpha[2],
                    'beta': beta,
                    'gamma': gamma,
                    'skew': skew,
                    'potential': potential
                   })

    elif spde_type == "heat":
        parameter_df = pd.DataFrame({'spde_type': [spde_type],
                    'diffusion': diffusion_coefficient,
                    'noise_type': noise_type,
                    'noise_coefficient': noise_coefficient,
                    'covariance_radius': covariance_radius,
                    'covariance_decay': covariance_decay
                   })

    parameter_df.to_csv(os.path.join(working_dir, 'signal', 'signal_parameters.csv'), index = False)

###############################################################################################################################################################################

    
def load_spde_parameters(working_dir):

    parameters_df = pd.read_csv(os.path.join(working_dir, 'signal', 'signal_parameters.csv'))
    
    # if spde_type == fitzhugh_nagumo     
    # returns in following order: [spde_type, [0: diffusion_act, 1: noise_type_act, 2: noise_coefficient_act, 3: covariance_radius_act, 4: covariance_decay_act, 5: diffusion_inh,   
    #                             6: noise_type_inh, 7: noise_coefficient_inh, 8: covariance_radius_inh, 9: covariance_decay_inh, 10: alpha_1, 11: alpha_2, 12: alpha_3, 13: beta, 14: gamma, 15: skew, 16: potential]]
    # elif spde_type == heat
    # returns in following order: [spde_type,  [0: diffusion, 1: noise_type, 2: noise_coefficient, 3: covariance_radius, 4: covariance_decay]]


    return  [parameters_df.values[0,0] ,parameters_df.values[0,1:]]
    


def load_spde_type(working_dir):
    return load_spde_parameters(working_dir)[0]


###############################################################################################################################################################################

# This function returns the general forward step type for heat or FitzHugh-Nagumo dynamics

def return_forward_step_type(signal_type):

    if signal_type == "heat":
        return spde_2d_simulations.forward_step_standard


    elif signal_type == "fitzhugh_nagumo":
        return spde_2d_simulations.forward_step_fhn
    
############################################################################################################################################################################################################### 

# This function returns signal forward step functions needed in the filtering algorthims

def return_forward_step_functions(signal_type, noise_type, noise_type_inh = 0):


    # for heat equation type
    if signal_type == "heat":
        
        forward_step_functions = [spde_2d_simulations.forward_linear_operator]  
        
        if noise_type == "add_white":
            forward_step_functions.append(spde_2d_simulations.forward_add_noise)
            
        elif noise_type == "add_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_add_correlated_noise)

        elif noise_type == "mult_white":
            forward_step_functions.append(spde_2d_simulations.forward_mul_noise_coefficient)
            
        elif noise_type == "mult_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_mul_correlated_noise_coefficient)
        
        
        return np.array(forward_step_functions)
    
    
    
    
    # for FithHugh-Nagumo type
    elif signal_type == "fitzhugh_nagumo":
        
        forward_step_functions = [spde_2d_simulations.forward_fhn_activator]    
                
        if noise_type == "add_white":
            forward_step_functions.append(spde_2d_simulations.forward_add_noise)
            
        elif noise_type == "add_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_add_correlated_noise)

        elif noise_type == "mult_white":
            forward_step_functions.append(spde_2d_simulations.forward_mul_noise_coefficient)
            
        elif noise_type == "mult_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_mul_correlated_noise_coefficient)
            

        forward_step_functions.append(spde_2d_simulations.forward_fhn_inhibitor)
        
        
        if noise_type_inh == "add_white":
            forward_step_functions.append(spde_2d_simulations.forward_add_noise)
            
        elif noise_type_inh == "add_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_add_correlated_noise)

        elif noise_type_inh == "mult_white":
            forward_step_functions.append(spde_2d_simulations.forward_mul_noise_coefficient)
            
        elif noise_type_inh == "mult_coloured":
            forward_step_functions.append(spde_2d_simulations.forward_mul_correlated_noise_coefficient)
        
        
        return np.array(forward_step_functions)
    

###############################################################################################################################################################################


# this functions returns the function arguments for the noise functions in the SPDE simulation

def return_forward_step_noise_args(spde_type, spde_parameters, dx, dim_x, dim_y):

    
    # for heat equation type
    if spde_type == "heat":
        
        noise_type = spde_parameters[1]
        
        noise_parameter_fem = spde_parameters[2]/dx
    

        # white noise
        if noise_type == "add_white" or noise_type == "mult_white":
            noise_args = [noise_parameter_fem]
        
        # colored noise using sqrt_Q
        elif noise_type == "add_coloured" or noise_type == "mult_coloured":
            # returns the sqrt of a simple spatial local covariance matrix
            sqrt_Q = spde_2d_simulations.return_simple_covariance_matrix(spde_parameters[3], spde_parameters[4], dim_x, dim_y, return_sqrt= True)
            noise_args = [noise_parameter_fem, sqrt_Q]
    
        return noise_args

    
    
    # for FithHugh-Nagumo type
    elif spde_type == "fitzhugh_nagumo":
        

        # auxilary indicator
        noise_type_act = spde_parameters[1]
        noise_type_inh = spde_parameters[6]
        
        # FEM noise coefficient
        noise_parameter_fem_act = spde_parameters[2]/dx
        noise_parameter_fem_inh = spde_parameters[7]/dx

        # for activator spde
        # white noise
        if noise_type_act == "add_white" or noise_type_act == "mult_white":
            noise_args_act = [noise_parameter_fem_act]
        
        # colored noise using sqrt_Q
        elif noise_type_act == "add_coloured" or noise_type_act == "mult_coloured":
            # returns the sqrt of a simple spatial local covariance matrix
            sqrt_Q_act = spde_2d_simulations.return_simple_covariance_matrix(spde_parameters[3], spde_parameters[4], dim_x, dim_y, return_sqrt= True)
            noise_args_act = [noise_parameter_fem_act, sqrt_Q_act]
            

        # for inhibitor spde
        # white noise
        if noise_type_inh == "add_white" or noise_type_inh == "mult_white":
            noise_args_inh = [noise_parameter_fem_inh]
        
        # colored noise using sqrt_Q
        elif noise_type_inh == "add_coloured" or  noise_type_inh == "mult_coloured":
            # returns the sqrt of a simple spatial local covariance matrix
            sqrt_Q_inh = spde_2d_simulations.return_simple_covariance_matrix(spde_parameters[8], spde_parameters[9], dim_x, dim_y, return_sqrt= True)
            noise_args_inh  = [noise_parameter_fem_inh, sqrt_Q_inh]


        return (noise_args_act, noise_args_inh)




###############################################################################################################################################################################


def save_initial_condition(working_dir, spde_type, initital_condition):

    if spde_type == "heat":
        np.save(os.path.join(working_dir, 'signal', "initial_condition.npy"), initital_condition)

    elif spde_type == "fitzhugh_nagumo":
        np.save(os.path.join(working_dir, 'signal', "initial_condition_activator.npy"), initital_condition[0])
        np.save(os.path.join(working_dir, 'signal', "initial_condition_inhibitor.npy"), initital_condition[1])


def load_initial_condition(working_dir, spde_type):

    if spde_type == "heat":
        return    np.load(os.path.join(working_dir, 'signal', "initial_condition.npy"))


    elif spde_type == "fitzhugh_nagumo":
        return    (np.load(os.path.join(working_dir, 'signal', "initial_condition_activator.npy")), np.load(os.path.join(working_dir, 'signal', "initial_condition_inhibitor.npy")))
    


###############################################################################################################################################################################


def return_spde_simulation_args(working_dir, dt, dx, dim_x, dim_y, eulerstep_type):

    spde_type, spde_parameters = load_spde_parameters(working_dir)


    # forward_step_type for Euler steps -> system or single SPDE
    forward_step_type = return_forward_step_type(spde_type)


    if spde_type == "fitzhugh_nagumo":                   


            # forward_step functions for Euler steps
            noise_type_act = spde_parameters[1]
            noise_type_inh = spde_parameters[6]

            forward_step_functions = return_forward_step_functions(spde_type, noise_type_act, noise_type_inh)

    # if spde_type == fitzhugh_nagumo     
    # returns in following order: [spde_type, [0: diffusion_act, 1: noise_type_act, 2: noise_coefficient_act, 3: covariance_radius_act, 4: covariance_decay_act, 5: diffusion_inh,   
    #                             6: noise_type_inh, 7: noise_coefficient_inh, 8: covariance_radius_inh, 9: covariance_decay_inh, 10: alpha_1, 11: alpha_2, 12: alpha_3, 13: beta, 14: gamma, 15: skew, 16: potential]]
    # elif spde_type == heat
    # returns in following order: [spde_type,  [0: diffusion, 1: noise_type, 2: noise_coefficient, 3: covariance_radius, 4: covariance_decay]]

            #Laplacians for explicit or implicit Euler
            if eulerstep_type == "explicit":
                
                # FEM diffusion coefficient
                diffusion_coeff_fem_inh = spde_parameters[0]/dx**2
                diffusion_coeff_fem_act = spde_parameters[5]/dx**2
                
                # 2d_Neumann Laplacians 
                A_inh = spde_2d_simulations.calc_neumann_laplacian_2d(dim_x, dim_y, (-1)*diffusion_coeff_fem_inh)
                A_act = spde_2d_simulations.calc_neumann_laplacian_2d(dim_x, dim_y, (-1)*diffusion_coeff_fem_act)

            elif eulerstep_type == "implicit":
                I_A_act, I_A_inh = 0                 

            

            # These lists contain the parameters that will be used in the simulation function
            # drift functions
            forward_step_args_act_drift =  [A_act, [ spde_parameters[10],  spde_parameters[11],  spde_parameters[12] ],  spde_parameters[15],  spde_parameters[16]]
            forward_step_args_inh_drift =  [A_inh, spde_parameters[13],  spde_parameters[14]]
            
            #noise_functions
            forward_step_args_act_noise, forward_step_args_inh_noise = return_forward_step_noise_args(spde_type, spde_parameters, dx, dim_x, dim_y)

            # final list
            forward_step_functions_args = [forward_step_args_act_drift, forward_step_args_act_noise, forward_step_args_inh_drift, forward_step_args_inh_noise]

    
    elif spde_type == "heat":
   
            # forward_step functions for Euler steps
            noise_type = spde_parameters[1]

            forward_step_functions = return_forward_step_functions(spde_type, noise_type)

            #Laplacians for explicit or implicit Euler
            if eulerstep_type == "explicit":
                
                # FEM diffusion coefficient
                diffusion_coeff_fem = spde_parameters[0]/dx**2
                
                # 2d_Neumann Laplacians 
                A = spde_2d_simulations.calc_neumann_laplacian_2d(dim_x, dim_y, (-1)*diffusion_coeff_fem)
                
            elif eulerstep_type == "implicit":
                I_A = 0                 

            

            # These lists contain the parameters that will be used in the simulation function
            # drift functions
            forward_step_args_drift =  [A]
            
            
            #noise_functions
            forward_step_args_noise = return_forward_step_noise_args(spde_type, spde_parameters, dx, dim_x, dim_y)

            # final list
            forward_step_functions_args = [forward_step_args_drift, forward_step_args_noise]
                        
    


    # load the initial condition
    initial_condition = load_initial_condition(working_dir, spde_type)
    

    # Euler step parameter for simulation function
    if eulerstep_type == "explicit":
        implicit_bool = False
    elif eulerstep_type == "implicit":
        implicit_bool = True
    
    
    ### need to change this
    if spde_type == "heat":
         simulation_type = "standard"
    elif spde_type == "fitzhugh_nagumo":
         simulation_type = "act_inh"


    return (simulation_type,  implicit_bool, initial_condition, forward_step_type, forward_step_functions, forward_step_functions_args)

###############################################################################################################################################################################

# simulates an SPDE path from .csv and .npy data in load_path

def simulate_2d_spde_path_from_csv(working_dir, print_steps = False, save_simulation = False):


    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = load_spde_metadata(working_dir)

    dim_x_y = dim_x*dim_y

    simulation_type,  implicit_bool, initial_condition, forward_step_type, forward_step_functions, forward_step_functions_args = return_spde_simulation_args(working_dir, dt, dx, dim_x, dim_y, eulerstep_type)

    

    if save_simulation == False:
        return spde_2d_simulations.simulate_signal_path(dt, timesteps, dim_x_y,  initial_condition, forward_step_type, forward_step_functions, forward_step_functions_args, type = simulation_type, implicit = implicit_bool, time_homog = True, print_steps = print_steps)

    else:
        simulation_path = spde_2d_simulations.simulate_signal_path(dt, timesteps, dim_x_y,  initial_condition, forward_step_type, forward_step_functions, forward_step_functions_args, type = simulation_type, implicit = implicit_bool, time_homog = True, print_steps = print_steps)
        
        spde_type = load_spde_type(working_dir)
        save_spde_path(working_dir, simulation_path, spde_type = spde_type)

        return simulation_path

###############################################################################################################################################################################

# input np.array([timesteps, dim_x*dim_y]), output np.array([timesteps, dim_x, dim_y])

def return_path_for_2d_plotting(path, timesteps, dim_x, dim_y):
    return path.reshape([timesteps, dim_x, dim_y])


###############################################################################################################################################################################

def save_spde_path(working_dir, spde_path, spde_type = "heat"):
    
    if spde_type == "heat":
         np.save(os.path.join(working_dir,'signal', "spde_path.npy"), spde_path)

    elif spde_type == "fitzhugh_nagumo":
         np.save(os.path.join(working_dir,'signal', "activator_path.npy"), spde_path[0])
         np.save(os.path.join(working_dir,'signal', "inhibitor_path.npy"), spde_path[1])


def load_spde_path(working_dir,  spde_type = "heat"):
    
    if spde_type == "heat":
         return np.load(os.path.join(working_dir,'signal', "spde_path.npy"))
    
    elif spde_type == "fitzhugh_nagumo":
         return (np.load(os.path.join(working_dir,'signal', "activator_path.npy")), np.load(os.path.join(working_dir,'signal', "inhibitor_path.npy")))
    

###############################################################################################################################################################################

