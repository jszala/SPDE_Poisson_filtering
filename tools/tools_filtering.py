import numpy as np
import pandas as pd
import os
from shutil import copyfile




import tools.tools_spde as tools_spde
import tools.tools_os as tools_os
import tools.tools_mpp as tools_mpp

import filtering.filter_module as filter_module
import filtering.observations_module as observations_module

import spde.spde_2d_simulations as spde_2d_simulations


###############################################################################################################################################################################

# Auxiallary functions to load observation data

###############################################################################################################################################################################


# This function searches for all observations in the working directory and returns a dictionary with all observation directories for further use

def return_observation_dirs_as_dict(working_dir):
    
    if os.path.isdir(os.path.join(working_dir, "observations")):
        
        if os.path.isfile(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv'))):
            
            directory_dict = {}

            observation_ids = pd.read_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')))['observation_id'].values
            
            for observation_id in observation_ids:

                # save observation_dir for the original observation to the directory_dict
                directory_dict[observation_id] = tools_os.return_observation_id_folder(working_dir, observation_id, create_dir = False)


                # check if downscaled observations exist
                if os.path.isdir(os.path.join(os.path.join(working_dir, "observations", observation_id, "downscaled"))):

                    downscaling_df_temp = pd.read_csv(os.path.join(os.path.join(working_dir, "observations", observation_id, "downscaled", 'downscaling_ids.csv')))
                    
                    for index in downscaling_df_temp.index:
                        
                        downscaling_factor_temp = downscaling_df_temp["downscaling_factor"].loc[index]
                        downscaling_type_temp = downscaling_df_temp["downscaling_type"].loc[index]
                        

                        if downscaling_type_temp == "sum":
                            
                            partial_method_temp = 0
                            downscaling_id = observation_id+"_"+str(downscaling_factor_temp)+"_"+downscaling_type_temp

                            
                        elif downscaling_type_temp == "partial":

                            partial_method_temp = downscaling_df_temp["partial_downscaling_method"].loc[index]
                            downscaling_id = observation_id+"_"+str(downscaling_factor_temp)+"_"+downscaling_type_temp+"_"+partial_method_temp

                        
                        directory_dict[downscaling_id] = tools_os.return_downscaled_directory(working_dir, observation_id, downscaling_factor_temp, downscaling_type_temp, partial_method_temp, create_dir = False)           
            
            # return dict
            return directory_dict

        else:
            print("observation_ids.csv: File not found.")
            return

    else:
        print("No observation directory found.")
        return


###############################################################################################################################################################################

# reads out the observation_metadata.csv in the observation directory and returns the values
def load_observation_metadata_from_dir(observation_dir):

    parameters_df = pd.read_csv(os.path.join(observation_dir,  'observation_metadata.csv'))
    
    return parameters_df.values[0,:]


# if the observation directory contains "downscaling_metadata.csv", then this function can be used to read the csv and return the values
def load_downscaled_observation_metadata_from_dir(observation_dir):
        if os.path.isfile(os.path.join(os.path.join(observation_dir, "downscaling_metadata.csv"))) == True:
                downscaling_df = pd.read_csv(os.path.join(observation_dir, "downscaling_metadata.csv" ))

        # returns downscaling_type, partial_method
        return downscaling_df.values[0,:]



# returns weight_function and weight_function_args for particle filter, i.e. the Poisson Radon-Nikodym density and the corresponding dx/dt values
def return_poisson_weight_function_for_particle_filter(observation_dir):
    
    weight_function = filter_module.poisson_weight_function
    

    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_from_dir(observation_dir)
    weight_function_args = [(dx_obs)**2, dt_obs]
    
    
    return (weight_function, weight_function_args)


# returns intensity function and parameters for filtering
def load_observation_intensity_parameters_from_dir(observation_dir):

    parameters_df = pd.read_csv(os.path.join(observation_dir, 'intensity_parameters.csv'))

    intensity_type = parameters_df['intensity_type'][0]

    if intensity_type ==  "linear" :
        return  [intensity_type, [parameters_df['multiplicator'][0]]]
    elif intensity_type ==  "exponent" :
        return [intensity_type, [parameters_df['multiplicator'][0], parameters_df['exponent'][0]]]
    elif intensity_type ==  "inhom_linear" :
        return [intensity_type, [parameters_df['multiplicator'][0], parameters_df['decay'][0]]]
    elif intensity_type ==  "inhom_exponent" :
        return (intensity_type, [parameters_df['multiplicator'][0], parameters_df['exponent'][0], parameters_df['decay'][0]])


###############################################################################################################################################################################

# Auxiliary function that returns observation_func and observation_func_args for particle filter
# for downscaled observations the corresponding identifiers have to be passed
# otherwise the function uses the standard observation path corr. to observation_id

def return_observation_function_for_particle_filter(observation_dir):

    # load_observation_parameters
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_from_dir(observation_dir)

    # load intensity parameters from intensity_parameters.csv
    intensity_type, intensity_parameters = load_observation_intensity_parameters_from_dir(observation_dir)

    observation_function = tools_mpp.return_intensity_function(intensity_type)




    # this case is for the original observation
    if spatial_multiplier == 1:
    
        observation_function_args = tools_mpp.return_intensity_function_args(intensity_parameters, 1, int(dim_x_obs*spatial_multiplier), int(dim_y_obs*spatial_multiplier), dx_obs)
        
        return (observation_function, observation_function_args)
    
    
    # this case is for the downscaled versions
    else:
        
        downscaling_type, partial_method = load_downscaled_observation_metadata_from_dir(observation_dir)

        # depending on if we want partial observations we need different matrices
        if downscaling_type == "sum":
            downscaling_matrix = observations_module.return_downscaling_matrix(int(spatial_multiplier), int(dim_x_obs*spatial_multiplier), int(dim_y_obs*spatial_multiplier), type = "integration", dx_old= dx_obs/spatial_multiplier)
        
        elif downscaling_type == "partial":
            downscaling_matrix = observations_module.return_downscaling_matrix(int(spatial_multiplier), int(dim_x_obs*spatial_multiplier), int(dim_y_obs*spatial_multiplier), type = "partial", partial_method= partial_method,   dx_old= dx_obs/spatial_multiplier)   
        

        return (observation_function, [downscaling_matrix, *intensity_parameters])


###############################################################################################################################################################################

# loads observation_path.npy from observation directory

def load_observation_path_from_dir(observation_dir):

    return np.load(os.path.join(observation_dir, 'observation_path.npy'))


###############################################################################################################################################################################

# Loads the signal path from working_dir. 

def load_signal_path(working_dir):

    spde_type = tools_spde.load_spde_type(working_dir)
    
    if spde_type == "heat":
         return tools_spde.load_spde_path(working_dir,  spde_type = spde_type)
    
    # in case of fitzhugh_nagumo return activator path
    elif spde_type == "fitzhugh_nagumo":
         return tools_spde.load_spde_path(working_dir,  spde_type = spde_type)[0]




###############################################################################################################################################################################

# returns signal parameters for filtering algorithm

def return_signal_args_for_filtering(working_dir):
    
    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = tools_spde.load_spde_metadata(working_dir)
    
    simulation_type,  implicit_bool, initial_condition_real, forward_step_type, forward_step_functions, forward_step_functions_args = tools_spde.return_spde_simulation_args(working_dir, dt, dx, dim_x, dim_y, eulerstep_type)
    '''
    if simulation_type == "standard":
        initial_condition_variance = load_initial_condition_variance(working_dir, "heat")
    elif simulation_type == "act_inh":
         initial_condition_variance = load_initial_condition_variance(working_dir, "fitzhugh_nagumo")

    print(initial_condition_variance)
   '''
    
    initial_condition, initial_condition_covariance = load_signal_initital_condition_args(working_dir)
    

    return [simulation_type,  implicit_bool, initial_condition, initial_condition_covariance, forward_step_type, forward_step_functions, forward_step_functions_args]



###############################################################################################################################################################################

# this function uses all csv data in the working directory tree to simulate the signal and the observations

def simulate_and_save_signal_and_observations_in_working_dir(working_dir):

    tools_spde.simulate_2d_spde_path_from_csv(working_dir, print_steps = True, save_simulation= True)
    
    observation_ids = pd.read_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')))['observation_id'].values
            
    for observation_id in observation_ids:

        tools_mpp.compute_intensity_from_signal(working_dir, observation_id, save_simulation = True)
        tools_mpp.simulate_cox_process_path(working_dir, observation_id, shift_for_filter = True, save_simulation = True)    
        tools_mpp.compute_spatially_downscaled_observations_from_batch_csv(working_dir, observation_id, save_simulations = True)
        

    return


###############################################################################################################################################################################

# saves/loads the initial condition mean and covariance matrix for initalizing the filter
def save_signal_initital_condition_args(working_dir, initital_condition_mean, intitial_condition_cov_matrix):
    

    spde_type = tools_spde.load_spde_type(working_dir)

    if spde_type == "heat":
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_mean.npy'), initital_condition_mean)
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_cov_matrix.npy'), intitial_condition_cov_matrix)
    
    elif spde_type == "fitzhugh_nagumo":
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_act_mean.npy'), initital_condition_mean[0])
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_act_cov_matrix.npy'), intitial_condition_cov_matrix[0])
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_inh_mean.npy'), initital_condition_mean[1])
        np.save(os.path.join(working_dir, 'filtering', 'initial_condition_inh_cov_matrix.npy'), intitial_condition_cov_matrix[1])

    return



def load_signal_initital_condition_args(working_dir):
    
    spde_type = tools_spde.load_spde_type(working_dir)

    if spde_type == "heat":
        return (np.load(os.path.join(working_dir, 'filtering', 'initial_condition_mean.npy')), np.load(os.path.join(working_dir, 'filtering', 'initial_condition_cov_matrix.npy')))
    
    elif spde_type == "fitzhugh_nagumo":
        return [[np.load(os.path.join(working_dir, 'filtering', 'initial_condition_act_mean.npy')), np.load(os.path.join(working_dir, 'filtering', 'initial_condition_inh_mean.npy'))],
                [np.load(os.path.join(working_dir, 'filtering', 'initial_condition_act_cov_matrix.npy')), np.load(os.path.join(working_dir, 'filtering', 'initial_condition_inh_cov_matrix.npy'))]]
    
###############################################################################################################################################################################

def save_particle_filter_estimate(estimate_path, working_dir, observation_id, estimator_id, ensemble_size):


    estimator_observation_id_dir  =  tools_os.return_estimator_observation_id_dir(working_dir, estimator_id, observation_id, create_dir = False)

    if os.path.isdir(estimator_observation_id_dir) == False:
        tools_os.return_estimator_observation_id_dir(working_dir, estimator_id, observation_id, create_dir = True)


    estimator_id_dir = tools_os.return_estimator_id_dir(working_dir, estimator_id, create_dir = False)
    # for automated reading
    ## if csv data is not yet created
    if os.path.isfile(os.path.join(estimator_id_dir, 'estimates.csv')) == False:
        

        estimate_df = pd.DataFrame({'observation_id': [str(observation_id)],
                                    'ensemble_size': ensemble_size                                          
            })
        
        # save observation_id to csv for automated reading
        estimate_df.to_csv(os.path.join(estimator_id_dir, 'estimates.csv'), index = False)

    
    else:            
        #load existing csv
        estimate_df = pd.read_csv(os.path.join(estimator_id_dir, 'estimates.csv'))

        # Check if row already exists. If it exists do nothing.
        if estimate_df.loc[(estimate_df['observation_id']==observation_id)&(estimate_df['ensemble_size']==ensemble_size)].any().all() == False:
            
            # If not, append downscaling info to existing csv
            new_row_index = estimate_df.shape[0]
            estimate_df.loc[new_row_index] = [str(observation_id), ensemble_size ]

            # save downscaling_info to csv for automated reading
            estimate_df.to_csv(os.path.join(estimator_id_dir, 'estimates.csv'), index = False)

    
    estimate_filename = "estimate_ensemblesize_"+str(ensemble_size)+".npy"

    np.save(os.path.join(estimator_observation_id_dir, estimate_filename),estimate_path)

    
###############################################################################################################################################################################


def run_particle_filter_with_estimator_id(working_dir, observation_dir, estimator_id, ensemble_size, print_steps = True, save_simulation = False):
    
    # load signal parameters
    signal_param_temp = return_signal_args_for_filtering(working_dir, estimator_id)

    # standard or act_inh
    signal_type = signal_param_temp[0]
    
    # explicit or implicit Euler
    implicit_bool = signal_param_temp[1]
    
    # signal specific arguments, such as initial condition for filter, Laplacian etc.
    signal_args = signal_param_temp[2:]

    
    # load observation metadata
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_from_dir(observation_dir)

    timesteps_obs = int(timesteps_obs)
    dim_x_obs = int(dim_x_obs)
    dim_y_obs = int(dim_y_obs)


    # load observation path
    observation_path = load_observation_path_from_dir(observation_dir)

    # load observation forward step function and arguments
    observation_func, observation_func_args = return_observation_function_for_particle_filter(observation_dir)
    
    # load Poisson radon-nikodym densities as weight functions for particle filter
    weight_fct, weight_fct_args = return_poisson_weight_function_for_particle_filter(observation_dir)


    if save_simulation == True:
        estimate = filter_module.particle_filter_estimate(observation_path, ensemble_size, dt_obs, time_multiplier, signal_type, signal_args, 
               
                                                          observation_func, observation_func_args, weight_fct, weight_fct_args, static_coefficients = True, print_steps = print_steps, implicit = implicit_bool)
        
        # need observation_id for saving in the correct directory
        observation_id = observation_dir.split("/")[-1]

        save_particle_filter_estimate(estimate, working_dir, observation_id, estimator_id, ensemble_size)

        return estimate
    
    else:
        return filter_module.particle_filter_estimate(observation_path, ensemble_size, dt_obs, time_multiplier, signal_type, signal_args, 
                                                      observation_func, observation_func_args, weight_fct, weight_fct_args, static_coefficients = True, print_steps = print_steps, implicit = implicit_bool)
    
###############################################################################################################################################################################
# saves/loads the initial condition mean and covariance matrix for initalizing the filter
def save_signal_initital_condition_args(working_dir, estimator_id, initital_condition_mean, intitial_condition_cov_matrix):
    
    estimator_id_dir = os.path.join(working_dir, 'filtering', estimator_id)

    if os.path.isdir(estimator_id_dir) == False:
        tools_os.return_estimator_id_dir(working_dir, estimator_id, create_dir = True)

    spde_type = load_spde_type_for_filtering(working_dir, estimator_id)

    if spde_type == "heat":
        np.save(os.path.join(estimator_id_dir, 'initial_condition_mean.npy'), initital_condition_mean)
        np.save(os.path.join(estimator_id_dir, 'initial_condition_cov_matrix.npy'), intitial_condition_cov_matrix)
    
    elif spde_type == "fitzhugh_nagumo":
        np.save(os.path.join(estimator_id_dir, 'initial_condition_act_mean.npy'), initital_condition_mean[0])
        np.save(os.path.join(estimator_id_dir, 'initial_condition_act_cov_matrix.npy'), intitial_condition_cov_matrix[0])
        np.save(os.path.join(estimator_id_dir, 'initial_condition_inh_mean.npy'), initital_condition_mean[1])
        np.save(os.path.join(estimator_id_dir, 'initial_condition_inh_cov_matrix.npy'), intitial_condition_cov_matrix[1])

    return



def load_signal_initital_condition_args(working_dir, estimator_id):
    
    spde_type = load_spde_type_for_filtering(working_dir, estimator_id)

    if spde_type == "heat":
        return (np.load(os.path.join(working_dir, 'filtering',  estimator_id, 'initial_condition_mean.npy')), np.load(os.path.join(working_dir, 'filtering',  estimator_id, 'initial_condition_cov_matrix.npy')))
    
    elif spde_type == "fitzhugh_nagumo":
        return [[np.load(os.path.join(working_dir, 'filtering', estimator_id, 'initial_condition_act_mean.npy')), np.load(os.path.join(working_dir, 'filtering', estimator_id, 'initial_condition_inh_mean.npy'))],
                [np.load(os.path.join(working_dir, 'filtering', estimator_id, 'initial_condition_act_cov_matrix.npy')), np.load(os.path.join(working_dir, 'filtering', estimator_id, 'initial_condition_inh_cov_matrix.npy'))]]


###############################################################################################################################################################################

# These functions are versions of the ones in tools_spde, but use the filtering signal model parameters

def load_spde_parameters_for_filtering(working_dir, estimator_id):

    parameters_df = pd.read_csv(os.path.join(working_dir, 'filtering', estimator_id, 'signal_parameters.csv'))
    
    # if spde_type == fitzhugh_nagumo     
    # returns in following order: [spde_type, [0: diffusion_act, 1: noise_type_act, 2: noise_coefficient_act, 3: covariance_radius_act, 4: covariance_decay_act, 5: diffusion_inh,   
    #                             6: noise_type_inh, 7: noise_coefficient_inh, 8: covariance_radius_inh, 9: covariance_decay_inh, 10: alpha_1, 11: alpha_2, 12: alpha_3, 13: beta, 14: gamma, 15: skew, 16: potential]]
    # elif spde_type == heat
    # returns in following order: [spde_type,  [0: diffusion, 1: noise_type, 2: noise_coefficient, 3: covariance_radius, 4: covariance_decay]]


    return  [parameters_df.values[0,0] ,parameters_df.values[0,1:]]
    


def load_spde_type_for_filtering(working_dir, estimator_id):
    return load_spde_parameters_for_filtering(working_dir, estimator_id)[0]


###############################################################################################################################################################################



# This function is a version of the one in tools_spde, but uses the filtering signal model parameters

def return_spde_simulation_args_for_filtering(working_dir, estimator_id, dt, dx, dim_x, dim_y, eulerstep_type):

    spde_type, spde_parameters = load_spde_parameters_for_filtering(working_dir, estimator_id)


    # forward_step_type for Euler steps -> system or single SPDE
    forward_step_type = tools_spde.return_forward_step_type(spde_type)


    if spde_type == "fitzhugh_nagumo":                   


            # forward_step functions for Euler steps
            noise_type_act = spde_parameters[1]
            noise_type_inh = spde_parameters[6]

            forward_step_functions = tools_spde.return_forward_step_functions(spde_type, noise_type_act, noise_type_inh)

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
            forward_step_args_act_noise, forward_step_args_inh_noise = tools_spde.return_forward_step_noise_args(spde_type, spde_parameters, dx, dim_x, dim_y)

            # final list
            forward_step_functions_args = [forward_step_args_act_drift, forward_step_args_act_noise, forward_step_args_inh_drift, forward_step_args_inh_noise]

    
    elif spde_type == "heat":
   
            # forward_step functions for Euler steps
            noise_type = spde_parameters[1]

            forward_step_functions = tools_spde.return_forward_step_functions(spde_type, noise_type)

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
            forward_step_args_noise = tools_spde.return_forward_step_noise_args(spde_type, spde_parameters, dx, dim_x, dim_y)

            # final list
            forward_step_functions_args = [forward_step_args_drift, forward_step_args_noise]
                        
    


    # load the initial condition and covariance
    initial_condition, initial_condition_covariance = load_signal_initital_condition_args(working_dir, estimator_id)
    

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


    return (simulation_type,  implicit_bool, initial_condition, initial_condition_covariance, forward_step_type, forward_step_functions, forward_step_functions_args)

###############################################################################################################################################################################

def copy_spde_parameters_to_filtering_dir(working_dir, estimator_id):

    estimator_id_dir = os.path.join(working_dir, 'filtering', estimator_id)

    if os.path.isdir(estimator_id_dir) == False:
        tools_os.return_estimator_id_dir(working_dir, estimator_id, create_dir = True)

    copyfile(os.path.join(working_dir, 'signal', 'signal_parameters.csv'), os.path.join(estimator_id_dir, 'signal_parameters.csv'))
    

def return_signal_args_for_filtering(working_dir, estimator_id):
    
    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = tools_spde.load_spde_metadata(working_dir)
    
    #simulation_type,  implicit_bool, initial_condition_real, forward_step_type, forward_step_functions, forward_step_functions_args = tools_spde.return_spde_simulation_args(working_dir, dt, dx, dim_x, dim_y, eulerstep_type)

    simulation_type,  implicit_bool, initial_condition, initial_condition_covariance, forward_step_type, forward_step_functions, forward_step_functions_args = return_spde_simulation_args_for_filtering(working_dir, estimator_id, dt, dx, dim_x, dim_y, eulerstep_type)
    


    return [simulation_type,  implicit_bool, initial_condition, initial_condition_covariance, forward_step_type, forward_step_functions, forward_step_functions_args]

###############################################################################################################################################################################