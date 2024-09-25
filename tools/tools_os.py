import os
from shutil import copyfile
import pandas as pd


import tools.tools_spde as tools_spde
import tools.tools_mpp as tools_mpp

# returns the working_dir and creates the general parent tree structure

def return_working_dir_folders(working_dir_parent, working_dir_id, create_dir = False):

    if create_dir == True:
        os.makedirs(os.path.join(working_dir_parent, working_dir_id, "signal"), exist_ok= True)
        os.makedirs(os.path.join(working_dir_parent, working_dir_id, "observations"), exist_ok= True)
        os.makedirs(os.path.join(working_dir_parent, working_dir_id, "filtering"), exist_ok= True)


    return os.path.join(working_dir_parent, working_dir_id)

#####################################################################################################################################################################################################################

# returns/creates the observation_id directory


def return_observation_id_folder(working_dir, observation_id, create_dir = False):

    observation_id_dir = os.path.join(working_dir, "observations", observation_id)

    #update the observation_id csv
    if create_dir == True: 
        # for automated reading
        ## if csv data is not yet created
        if os.path.isfile(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv'))) == False:
            
            observation_id_df = pd.DataFrame({'observation_id': [str(observation_id)]
                })
            
            # save observation_id to csv for automated reading
            observation_id_df.to_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')), index = False)

        
        else:            
            
            #load existing csv
            observation_id_df = pd.read_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')))

            # Check if row already exists. If it exists do nothing.
            if observation_id_df.loc[observation_id_df['observation_id']==observation_id].any().all() == False:
                
                # If not, append downscaling info to existing csv
                new_row_index = observation_id_df.shape[0]
                observation_id_df.loc[new_row_index] = str(observation_id)

                # save downscaling_info to csv for automated reading
                observation_id_df.to_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')), index = False)
        

        # Option to create the directory
        os.makedirs(os.path.join(working_dir, "observations", observation_id), exist_ok= True)


    return observation_id_dir


#####################################################################################################################################################################################################################


# takes  downscaling_factor, downscaling_type, partial_downscaling_method as inputs and creates the correct directories and entries in the downscaling_ids.csv for automation

def return_downscaled_directory(working_dir, observation_id, downscaling_factor, downscaling_type, partial_downscaling_method = 0, create_dir = True):
    
    downscaled_dir_root = os.path.join(working_dir, "observations", observation_id, "downscaled")

    # This option creates/modifies the downscaling_ids.csv and creates the downscaled directory
    if create_dir == True:
        if downscaling_type == "sum":
            partial_downscaling_method = 0

        # if downscaled parent directory is not yet created
        if os.path.isdir(downscaled_dir_root) == False:
            os.makedirs(downscaled_dir_root)

        # saving downscaling factor for automated reading

        ## if csv data is not yet created
        if os.path.isfile(os.path.join(os.path.join(downscaled_dir_root, 'downscaling_ids.csv'))) == False:
            
            downscaling_df = pd.DataFrame({'downscaling_type': [downscaling_type],
                'downscaling_factor': downscaling_factor,
                'partial_downscaling_method': partial_downscaling_method
                })
            
            # save downscaling_info to csv for automated reading
            downscaling_df.to_csv(os.path.join(os.path.join(downscaled_dir_root, 'downscaling_ids.csv')), index = False)

        
        else:            

            downscaling_df = pd.read_csv(os.path.join(os.path.join(downscaled_dir_root, 'downscaling_ids.csv')))
            
            # Check if row already exists:
            if (downscaling_df.loc[(downscaling_df['downscaling_type']==downscaling_type)&(downscaling_df['downscaling_factor']==downscaling_factor)&(downscaling_df['partial_downscaling_method']==partial_downscaling_method)].any().all()) == False:
        
                # If not, append downscaling info to existing csv
                new_row_index = downscaling_df.shape[0]
                downscaling_df.loc[new_row_index] = [downscaling_type, downscaling_factor, partial_downscaling_method]

                # save downscaling_info to csv for automated reading
                downscaling_df.to_csv(os.path.join(os.path.join(downscaled_dir_root, 'downscaling_ids.csv')), index = False)

    

    
    # make directory for downscaled observation
    if downscaling_type == "sum":
        downscaled_path_dir = os.path.join(downscaled_dir_root, str(downscaling_factor)+"_"+downscaling_type)
       
    else:
        downscaled_path_dir = os.path.join(downscaled_dir_root, str(downscaling_factor)+"_"+downscaling_type+"_"+partial_downscaling_method)

    # Creates the downscaled observation directory
    if create_dir == True:

        # Check if directory already exists
        if os.path.isdir(downscaled_path_dir) == False:
            os.makedirs(downscaled_path_dir)


    return downscaled_path_dir

#####################################################################################################################################################################################################################


def return_estimator_id_dir(working_dir, estimator_id, create_dir = False):

    estimator_id_dir = os.path.join(working_dir, "filtering", estimator_id)

    #update the observation_id csv
    if create_dir == True: 
        # for automated reading
        ## if csv data is not yet created
        if os.path.isfile(os.path.join(os.path.join(working_dir, "filtering", 'estimator_ids.csv'))) == False:
            
            estimator_id_df = pd.DataFrame({'estimator_id': [str(estimator_id)]
                })
            
            # save observation_id to csv for automated reading
            estimator_id_df.to_csv(os.path.join(os.path.join(working_dir, "filtering", 'estimator_ids.csv' )), index = False)

        
        else:            
            
            #load existing csv
            estimator_id_df = pd.read_csv(os.path.join(os.path.join(working_dir, "filtering", 'estimator_ids.csv')))

            # Check if row already exists. If it exists do nothing.
            if estimator_id_df.loc[estimator_id_df['estimator_id']==estimator_id].any().all() == False:
                
                # If not, append downscaling info to existing csv
                new_row_index = estimator_id_df.shape[0]
                estimator_id_df.loc[new_row_index] = str(estimator_id)

                # save downscaling_info to csv for automated reading
                estimator_id_df.to_csv(os.path.join(os.path.join(working_dir, "filtering", 'estimator_ids.csv')), index = False)
        

        # Option to create the directory
        os.makedirs(estimator_id_dir, exist_ok= True)


    return estimator_id_dir


#####################################################################################################################################################################################################################

# this function creates the directory for the estimator for a specific observation
def return_estimator_observation_id_dir(working_dir, estimator_id, observation_id, create_dir = False):

    estimator_observation_id_dir = os.path.join(working_dir, "filtering", estimator_id, observation_id)

    if create_dir == True:
        
        if os.path.isdir(estimator_observation_id_dir) == False:
            os.makedirs(estimator_observation_id_dir)
        
       

    return estimator_observation_id_dir

#####################################################################################################################################################################################################################

# This function exports all the necessary csv data from a source directory to a new working directory.
# If export_from_folders == False, this function will simply copy the readied csv data. Otherwise it will search all folders and create the needed csv.

def export_all_signal_and_observation_csv_data(working_dir, destination_dir, export_from_folders = True):
    
    if export_from_folders == True:

        copyfile(os.path.join(working_dir,"signal", "signal_parameters.csv"), os.path.join(destination_dir, "signal_parameters.csv"))
        copyfile(os.path.join(working_dir,"signal", "metaparameters.csv"), os.path.join(destination_dir, "metaparameters.csv"))
        

        spde_type = tools_spde.load_spde_type(working_dir)

        if spde_type == "heat":
            copyfile(os.path.join(working_dir,"signal", "initial_condition.npy"), os.path.join(destination_dir,  "initial_condition.npy"))
        elif spde_type == "fitzhugh_nagumo":
            copyfile(os.path.join(working_dir,"signal", "initial_condition_activator.npy"), os.path.join(destination_dir,  "initial_condition_activator.npy"))
            copyfile(os.path.join(working_dir,"signal", "initial_condition_inhibitor.npy"), os.path.join(destination_dir,  "initial_condition_inhibitor.npy"))


        copyfile(os.path.join(working_dir,"observations",  'observation_ids.csv'), os.path.join(destination_dir, 'observation_ids.csv'))

        observation_metadata_all = pd.DataFrame(columns=['time_multiplier', 'spatial_multiplier'])
        intensity_parameters_all =  pd.DataFrame(columns=['intensity_type', 'multiplicator', 'exponent', 'decay'])

        

        observation_ids = pd.read_csv(os.path.join(os.path.join(working_dir, "observations", 'observation_ids.csv')))['observation_id'].values
                
        for observation_id in observation_ids:
            observation_metadata_all.loc[observation_id] = pd.read_csv(os.path.join(working_dir,"observations", observation_id, "observation_metadata.csv")).values[0,:2]
            intensity_parameters_all.loc[observation_id] = pd.read_csv(os.path.join(working_dir,"observations", observation_id, "intensity_parameters.csv")).values[0]
            
        observation_metadata_all.to_csv(os.path.join(destination_dir,  "observation_metadata_all.csv"))
        intensity_parameters_all.to_csv(os.path.join(destination_dir,  "intensity_parameters_all.csv"))
        
        if os.path.isdir(os.path.join(working_dir, "observations", observation_ids[0], "downscaled")) == True:
            copyfile(os.path.join(working_dir, "observations", observation_ids[0], "downscaled", "downscaling_ids.csv"), os.path.join(destination_dir,  "downscaling_ids.csv"))

    
    # if the above steps have already been done and if there is a source directory with all needed csv data, this case will simply copy the data to the new directory
    else:
        copyfile(os.path.join(working_dir,"signal_parameters.csv"), os.path.join(destination_dir, "signal_parameters.csv"))
        copyfile(os.path.join(working_dir, "metaparameters.csv"), os.path.join(destination_dir, "metaparameters.csv"))

        spde_type = pd.read_csv(os.path.join(working_dir, 'signal_parameters.csv')).values[0,0]

        
        if spde_type == "heat":
            copyfile(os.path.join(working_dir, "initial_condition.npy"), os.path.join(destination_dir,  "initial_condition.npy"))
        elif spde_type == "fitzhugh_nagumo":
            copyfile(os.path.join(working_dir, "initial_condition_activator.npy"), os.path.join(destination_dir,  "initial_condition_activator.npy"))
            copyfile(os.path.join(working_dir, "initial_condition_inhibitor.npy"), os.path.join(destination_dir,  "initial_condition_inhibitor.npy"))

        copyfile(os.path.join(working_dir, "observation_metadata_all.csv"), os.path.join(destination_dir,  "observation_metadata_all.csv"))
        copyfile(os.path.join(working_dir, "intensity_parameters_all.csv"), os.path.join(destination_dir,  "intensity_parameters_all.csv"))

        copyfile(os.path.join(working_dir,'observation_ids.csv'), os.path.join(destination_dir, 'observation_ids.csv'))
        
        if os.path.isfile(os.path.join(working_dir, "downscaling_ids.csv")):
            copyfile(os.path.join(working_dir, "downscaling_ids.csv"), os.path.join(destination_dir,  "downscaling_ids.csv"))



        


def create_working_dir_tree_and_csv_data_from_run_csv(working_dir):
    
    return_working_dir_folders(os.path.join("/",*working_dir.split("/")[:-1]), working_dir.split("/")[-1], create_dir= True)

    os.replace(os.path.join(working_dir, "signal_parameters.csv"), os.path.join(working_dir,"signal", "signal_parameters.csv"))
    os.replace(os.path.join(working_dir, "metaparameters.csv"), os.path.join(working_dir,"signal", "metaparameters.csv"))

    spde_type = tools_spde.load_spde_type(working_dir)

    if spde_type == "heat":
        os.replace(os.path.join(working_dir, "initial_condition.npy"),os.path.join(working_dir,"signal", "initial_condition.npy"))
    elif spde_type == "fitzhugh_nagumo":
        os.replace(os.path.join(working_dir, "initial_condition_activator.npy"), os.path.join(working_dir,"signal", "initial_condition_activator.npy"))
        os.replace(os.path.join(working_dir, "initial_condition_inhibitor.npy"), os.path.join(working_dir,"signal", "initial_condition_inhibitor.npy"))

    
    observation_metadata_all = pd.read_csv(os.path.join(working_dir,  "observation_metadata_all.csv"), index_col="Unnamed: 0")
    intensity_parameters_all = pd.read_csv(os.path.join(working_dir,  "intensity_parameters_all.csv"), index_col="Unnamed: 0")

    

    for observation_id in observation_metadata_all.index:

        os.makedirs(os.path.join(working_dir, "observations", observation_id), exist_ok=True)

        tools_mpp.save_observation_metadata_with_id(working_dir, observation_id, observation_metadata_all.loc[observation_id, "time_multiplier"], observation_metadata_all.loc[observation_id, "spatial_multiplier"])
        tools_mpp.save_observation_intensity_parameters(working_dir, observation_id, intensity_parameters_all.loc[observation_id, "intensity_type"], intensity_parameters_all.loc[observation_id, 'multiplicator'], intensity_parameters_all.loc[observation_id, 'exponent'],intensity_parameters_all.loc[observation_id,  'decay'])

        if os.path.isfile(os.path.join(working_dir,  "downscaling_ids.csv")) == True:
            os.makedirs(os.path.join(working_dir, "observations", observation_id, "downscaled"), exist_ok=True)
            copyfile( os.path.join(working_dir,  "downscaling_ids.csv"),  os.path.join(working_dir, "observations", observation_id, "downscaled", "downscaling_ids.csv"))            


    os.replace( os.path.join(working_dir, 'observation_ids.csv') , os.path.join(working_dir,"observations",  'observation_ids.csv'))

    os.remove(os.path.join(working_dir,  "observation_metadata_all.csv"))
    os.remove(os.path.join(working_dir,  "intensity_parameters_all.csv"))
    if os.path.isfile(os.path.join(working_dir,  "downscaling_ids.csv")) == True:
            os.remove(os.path.join(working_dir,  "downscaling_ids.csv"))


