import pandas as pd
import numpy as np
import os
from shutil import copyfile



import tools.tools_spde as tools_spde
import tools.tools_os as tools_os

import poisson_process.poisson_process_simulation as poisson_process_simulation

import filtering.observations_module as observations_module



##################################################################################################################################################################################################################
# Main observation tools #


def save_observation_metadata_with_id(working_dir, observation_id,  time_multiplier, spatial_multiplier):
    
    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = tools_spde.load_spde_metadata(working_dir)

    parameter_df = pd.DataFrame({'time_multiplier': [time_multiplier],
                'spatial_multiplier': spatial_multiplier,
                'timesteps_obs': timesteps*time_multiplier,
                'dt_obs': dt*time_multiplier,
                'dx_obs': dx*spatial_multiplier,
                'dim_x_obs': dim_x*spatial_multiplier,
                'dim_y_obs': dim_y*spatial_multiplier
                })

    parameter_df.to_csv(os.path.join(working_dir, 'observations', observation_id, 'observation_metadata.csv'), index = False)    



def load_observation_metadata_with_id(working_dir, observation_id):
    parameters_df = pd.read_csv(os.path.join(working_dir, 'observations', observation_id,  'observation_metadata.csv'))
    return parameters_df.values[0,:]

#####################################################################################################################################################################################################################

def save_observation_intensity_parameters(working_dir, observation_id, intensity_type, multiplicator, exponent = 1, decay = 0):
    

    parameter_df = pd.DataFrame({'intensity_type': [intensity_type],
                'multiplicator': multiplicator,
                'exponent': exponent,
                'decay': decay
                })

    parameter_df.to_csv(os.path.join(working_dir, 'observations', observation_id, 'intensity_parameters.csv'), index = False)


def load_observation_intensity_parameters(working_dir, observation_id):

    parameters_df = pd.read_csv(os.path.join(working_dir, 'observations', observation_id, 'intensity_parameters.csv'))

    intensity_type = parameters_df['intensity_type'][0]

    if intensity_type ==  "linear" :
        return  [intensity_type, [parameters_df['multiplicator'][0]]]
    elif intensity_type ==  "exponent" :
        return [intensity_type, [parameters_df['multiplicator'][0], parameters_df['exponent'][0]]]
    elif intensity_type ==  "inhom_linear" :
        return [intensity_type, [parameters_df['multiplicator'][0], parameters_df['decay'][0]]]
    elif intensity_type ==  "inhom_exponent" :
        return (intensity_type, [parameters_df['multiplicator'][0], parameters_df['exponent'][0], parameters_df['decay'][0]])


#####################################################################################################################################################################################################################


def return_intensity_function(intensity_type):
    # inhomogeneous with power
    if intensity_type ==  "linear" :
        return  poisson_process_simulation.hom_intensity_coefficient
    elif intensity_type ==  "exponent" :
        return poisson_process_simulation.hom_intensity_power
    elif intensity_type ==  "inhom_linear" :
        return poisson_process_simulation.inhom_intensity_exponential_coefficient
    elif intensity_type ==  "inhom_exponent" :
        return poisson_process_simulation.inhom_intensity_exponential_power



def return_intensity_function_args(intensity_parameters,spatial_multiplier, dim_x, dim_y, dx):

    integration_matrix =  observations_module.return_downscaling_matrix(spatial_multiplier, dim_x, dim_y, type = "integration", dx_old= dx)       

    return  [integration_matrix,  *intensity_parameters]


#####################################################################################################################################################################################################################

def save_observation_intensity_path(working_dir, observation_id, intensity_path):
    np.save(os.path.join(working_dir, 'observations', observation_id, 'intensity_path.npy'), intensity_path)

def load_observation_intensity_path(working_dir, observation_id):
    return  np.load(os.path.join(working_dir, 'observations', observation_id, 'intensity_path.npy'))

#####################################################################################################################################################################################################################


def compute_intensity_from_signal(working_dir, observation_id, save_simulation = False):
    # load signal metadata
    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = tools_spde.load_spde_metadata(working_dir)

    # load point process metadata
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_with_id(working_dir, observation_id)

    time_multiplier = int(time_multiplier)
    spatial_multiplier = int(spatial_multiplier)
    timesteps_obs = int(timesteps_obs)
    
    # load arguments for intensity function
    intensity_type, intensity_parameters = load_observation_intensity_parameters(working_dir, observation_id)

    
    intensity_function = return_intensity_function(intensity_type)

    intensity_function_args = return_intensity_function_args(intensity_parameters, spatial_multiplier, dim_x, dim_y, dx)

    
    spde_type = tools_spde.load_spde_parameters(working_dir)[0]

    if spde_type == "heat":
        signal_path = tools_spde.load_spde_path(working_dir,  spde_type = spde_type)
    elif spde_type == "fitzhugh_nagumo":
        signal_path = tools_spde.load_spde_path(working_dir,  spde_type = spde_type)[0]

    if save_simulation == False:
        return poisson_process_simulation.calculate_intensity_path(signal_path[::time_multiplier], dim_x*dim_y, timesteps_obs, dt_obs, intensity_function, intensity_function_args)

    else:
        intensity_path =  poisson_process_simulation.calculate_intensity_path(signal_path[::time_multiplier], dim_x*dim_y, timesteps_obs, dt_obs, intensity_function, intensity_function_args)
        save_observation_intensity_path(working_dir, observation_id, intensity_path)

        return intensity_path
    
 #####################################################################################################################################################################################################################

def save_cox_process_as_observation(working_dir, observation_id, cox_process_path):
    np.save(os.path.join(working_dir, 'observations', observation_id, 'observation_path.npy'), cox_process_path)

def load_cox_process_as_observation(working_dir, observation_id):
    return np.load(os.path.join(working_dir, 'observations', observation_id, 'observation_path.npy'))

#####################################################################################################################################################################################################################

def simulate_cox_process_path(working_dir, observation_id, shift_for_filter = True, save_simulation = False):
    
    # load point process metadata
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs = load_observation_metadata_with_id(working_dir, observation_id)
    intensity_path = load_observation_intensity_path(working_dir, observation_id)

    if save_simulation == False:
        return  poisson_process_simulation.simulate_cox_process_from_intensity(intensity_path, dt_obs, shift_for_filter= shift_for_filter)
    else:
        observation_path = poisson_process_simulation.simulate_cox_process_from_intensity(intensity_path, dt_obs, shift_for_filter= shift_for_filter)
        save_cox_process_as_observation(working_dir, observation_id, observation_path)
        return observation_path


##################################################################################################################################################################################################################





##################################################################################################################################################################################################################
##################################################################################################################################################################################################################

# Downscaled observation tools #

##################################################################################################################################################################################################################

# this function passes the information from downscaling_ids.csv into a list for automation purposes


def load_downscaling_batch_parameters(working_dir, observation_id):
    
        downscaled_dir_root = os.path.join(working_dir, "observations", observation_id, "downscaled")
        downscaling_df = pd.read_csv(os.path.join(os.path.join(downscaled_dir_root, 'downscaling_ids.csv')))
        
        number_of_rows = downscaling_df.index.shape[0]

        downscaling_types = [None]*number_of_rows
        downscaling_factors = [None]*number_of_rows
        partial_methods = [None]*number_of_rows

        k = 0
        
        for index in downscaling_df.index:

                downscaling_types[k], downscaling_factors[k],  partial_methods[k] = downscaling_df.loc[index].values[0], *downscaling_df.loc[index].values[1:]
                k += 1
                        

        return (downscaling_types, downscaling_factors, partial_methods)


##################################################################################################################################################################################################################


# save and load the metaparameters for downscaled observations. Uses the observation_metadata.csv in the observation directory as a reference
# this csv file is needed to correctly set up the filter algorithm

def save_downscaled_observation_metadata(working_dir, observation_id, downscaling_factor, downscaling_type, partial_downscaling_method = 0):
    
    # partial_downscaling_method = 0 for downscaling_type = "sum"
    
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_with_id(working_dir, observation_id)

    if downscaling_type == "sum":

        downscaled_observation_dir = os.path.join(working_dir, 'observations', observation_id, 'downscaled', str(downscaling_factor)+"_"+downscaling_type)

        parameter_df = pd.DataFrame({'time_multiplier': [time_multiplier],
                'spatial_multiplier': spatial_multiplier*downscaling_factor,
                'timesteps_obs': timesteps_obs,
                'dt_obs': dt_obs,
                'dx_obs': dx_obs*downscaling_factor,
                'dim_x_obs': dim_x_obs/downscaling_factor,
                'dim_y_obs': dim_y_obs/downscaling_factor
                })

        downscaling_df = pd.DataFrame({'downscaling_type': [downscaling_type],
                'partial_method': partial_downscaling_method,
                })
        
        
    
    else:
        
        downscaled_observation_dir = os.path.join(working_dir, 'observations', observation_id, 'downscaled', str(downscaling_factor)+"_"+downscaling_type+"_"+partial_downscaling_method)

        parameter_df = pd.DataFrame({'time_multiplier': [time_multiplier],
                'spatial_multiplier': spatial_multiplier*downscaling_factor,
                'timesteps_obs': timesteps_obs,
                'dt_obs': dt_obs,
                'dx_obs': dx_obs,
                'dim_x_obs': dim_x_obs/downscaling_factor,
                'dim_y_obs': dim_y_obs/downscaling_factor
                })

        downscaling_df = pd.DataFrame({'downscaling_type': [downscaling_type],
                'partial_method': partial_downscaling_method,
                })
        
        
    # save csv data    
    parameter_df.to_csv(os.path.join(downscaled_observation_dir, "observation_metadata.csv" ), index = False)
    downscaling_df.to_csv(os.path.join(downscaled_observation_dir, "downscaling_metadata.csv" ), index = False)
    
    # copy intensity parameters for automation in filtering
    copyfile(os.path.join(working_dir, 'observations', observation_id, 'intensity_parameters.csv'), os.path.join(downscaled_observation_dir, 'intensity_parameters.csv'))

    
    



##################################################################################################################################################################################################################

# auxiliary functions to save and load the downscaled observations in the correct directories



def save_downscaled_observation_path(working_dir, observation_id, downscaled_obs_path, downscaling_factor, downscaling_type, partial_downscaling_method = 0):                                                      

    downscaling_save_dir = tools_os.return_downscaled_directory(working_dir, observation_id, downscaling_factor, downscaling_type, partial_downscaling_method, create_dir= False)

    # check if the correct directory already exsists. If not, create it.
    if os.path.isdir(downscaling_save_dir) == False:
        tools_os.return_downscaled_directory(working_dir, observation_id, downscaling_factor, downscaling_type, partial_downscaling_method, create_dir= True)

    # save metadata
    save_downscaled_observation_metadata(working_dir, observation_id,downscaling_factor, downscaling_type, partial_downscaling_method)
    
    # save observation_path
    np.save(os.path.join(downscaling_save_dir, 'observation_path.npy'), downscaled_obs_path)



 
    
    
##################################################################################################################################################################################################################

# this function computes the downscaled observation given an observation directory and the three identifiers downscaling_factor, downscaling_type, partial_method. Optionally saves the downscaled version

def compute_spatially_downscaled_observation(working_dir, observation_id, downscaling_factor, downscaling_type, partial_method = 0, save_simulation = False):

    # load signal metadata
    dt, dx, timesteps, dim_x, dim_y, eulerstep_type = tools_spde.load_spde_metadata(working_dir)

    # load observation metadata
    time_multiplier, spatial_multiplier, timesteps_obs, dt_obs, dx_obs, dim_x_obs, dim_y_obs  = load_observation_metadata_with_id(working_dir, observation_id)
    
    dim_x = int(dim_x)
    dim_y = int(dim_y)
    timesteps_obs = int(timesteps_obs)
    
    observation_path = load_cox_process_as_observation(working_dir, observation_id)    

    dim_x_obs_downscaled = int(dim_x_obs/downscaling_factor)
    dim_y_obs_downscaled = int(dim_y_obs/downscaling_factor)

    dim_x_y_obs_downscaled = int(dim_x_obs_downscaled*dim_y_obs_downscaled)

    # depending on if we sum up pixel values or create partial observations by picking one pixel we need different downscaling matrices
    if downscaling_type == "sum":
        downscaling_matrix = observations_module.return_downscaling_matrix(downscaling_factor, dim_x, dim_y, type = downscaling_type)
    elif downscaling_type == "partial":
        downscaling_matrix = observations_module.return_downscaling_matrix(downscaling_factor, dim_x, dim_y, type = downscaling_type, dx_old= 1, partial_method = partial_method)
    
    downscaled_path = np.zeros(shape = (timesteps_obs, dim_x_y_obs_downscaled))

    for i in np.arange(0, timesteps_obs):
        downscaled_path[i] = np.matmul(downscaling_matrix, observation_path[i,:])

    if save_simulation == False:
        return downscaled_path
    
    else:
        save_downscaled_observation_path(working_dir, observation_id, downscaled_path, downscaling_factor, downscaling_type, partial_method)
        return downscaled_path
    
##################################################################################################################################################################################################################

# this function uses the downscaling_ids.csv file in the folder downscaled and returns all downscaled observations as a list with an option to save the paths

def compute_spatially_downscaled_observations_from_batch_csv(working_dir, observation_id, save_simulations = False):

    downscaling_types, downscaling_factors, partial_methods = load_downscaling_batch_parameters(working_dir, observation_id)

    downscaled_observations = []

    for k in np.arange(len(downscaling_types)):

        downscaled_observations.append(compute_spatially_downscaled_observation(working_dir, observation_id, downscaling_factors[k], downscaling_types[k], partial_methods[k], save_simulation = save_simulations))

    return downscaled_observations


##################################################################################################################################################################################################################