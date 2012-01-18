# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
====================================================================
    Making a Connectome Atlas with Tractography and QuickBundles 
===================================================================

                                                    JDG 10/01/2012  
                                    
Background:
-----------

Defines 1st and 2nd level workflows that combine tractography results
from a group of subjects into a single set of group-representative 
tracts. Unlike conventional 'tract-based' tractography clustering and 
atlas approaches, this method is 'connection-based': tractography 
streamlines are labelled according to which regions from a high-
resolution parcellation of the white matter surface they connect to. 

This analysis takes as input a series of .cff (connectome-file-format) 
files, generated from a separate analysis pipeline. The .cff files 
contain (amongst other things) the tractography streamlines and an array
that defines a label for each streamline, where each label consists of
two numbers - e.g. [201][409], which are the indices of the two regions
on the white matter surface that that streamline's endpoints fell within.

The basic functionality of this pipeline is to find the corresponding 
streamlines in each subject, transform them to standard space, and combine
them into a single group atlas. To improve the accuracy and efficiency of
this procedure, however, we use the 'QuickBundles' tractography clustering
algorithm to define representative streamlines at both the 1st (within-
subject) and 2nd (between subject) levels. 

Two types of representative streamline can be defined with QuickBundles: 
'exemplars' (E) and 'virtuals' (V). The exemplar is the element within a 
set of streamlines that is most representative of the set. The virtual 
is a 'synthetic' streamline (the 'medroid/centroid') that is not itself an 
element of the set but is most representative of the set. 

The atlas construction procedure is defined here for both Es and Vs (as 
well as the raw, non-clustered streamlines (Ts), for comparison. 

The aim of using QuickBundles at the 1st level is to account for the fact 
that there may be multiple streamlines identified as connecting a given 
pair of wm surface regions (especially if probabilistic tractography
is used). With fine-grained parcellation schemes the number of actual white 
matter structures captured for a given region pair is minimized, and so
typically there is a dominant set of streamlines that follow a similar 
trajectory, which is what we really want to capture when constructing our
atlas. However there are also frequently some 'outlier' streamlines that
take quite different trajectories to the majority (again, especially if
probabilstic tractography is used). QuickBundles at the 1st level serves 
the twin purpose of outlier removal and summarizing the dominant fibre 
structures. At the 2nd level the logic is much the same, in that we must 
allow for the fact that there will be individual variation in the precise
trajectory of a given connection, but (we expect) some dominant structure 
present in the group as a whole, and also that some connections in some 
subjects are identified more reliably than in other subjects, and (depending
on thresholding) some may effectively be 'noise'. 

The majority of the workflow nodes use custom nipype interfaces that I have 
written for the specific set of analysis tasks required here. Several of 
them are wrappers for QuickBundles, however QuickBundles itself was 
developed by friend and colleague E. Garyfallidis, and is part of the 
dipy software package - see nipy.sourceforge.net/dipy/


Outline:
--------

    1. Import modules and define variables

    2. Specify 1st level workflow:
   
            a.   Define datasource
            
            b.   Apply QuickBundles to connectome files
                - outputs as a .pkl file containing Ts, Es, Vs, labels, and
                  a few other things
 
            c.  Write out streamlines for each connection to trackvis format (.trk)
                files
          
            d. Apply affine diffusion space --> MNI space transform
                - To .pkl files
                - To .trk files (probably won't use)

            e. ?
            
            f: Calculate geometries of ROIs and brain size / volume
                
            h. Define workflow
        
    3. Run 1st level workflow

    4. Specify 2nd level workflow

            a. Define datasource
            
            b. Collect all subject's affine-transformed QuickBundles outputs
               and run a 2nd level QuickBundles clustering (done separately
               for Ts, Es, and Vs)
          
            c. Write out the results in .trk format (this is your atlas!) 
            
            d. Define workflow

    5. Run 2nd level workflow



To run, type

    >> python jg_make_cnxn_lengths_atlas_script.py
    
or

    >> ipython
    >> run jg_make_cnxn_lengths_atlas_script.py



  
Notes:
------


    - This website is a useful alternative to the nipype documentation - I used it for 
      the level 2 workflow: 
  
         http://miykael.github.com/nipype-beginner-s-guide/secondLevelExample.html

    - Q: Does the fact that QB clusters using lengths mean that even if it doesn't give 
        a very convincing geometry for the representative tracks, the length of them 
        will nevertheless be representative?
        
        
    - Still to do: 
    
        - inverse transforms mapping atlas back to native space (separate workflow?) 
        - Get 'write_graph' working!
        - Change the input / output field names in the various custom nipype interfaces
          used here to be more sensible
        - Get rid of all the unecessary terminal screen output, although maybe add in some useful
          processing update messages
        - Make sure the modification to the 'write_QB_to_trk' interface you did is added to 
          any other relevant interfaces (change = make the 'property' and 'scalar' fields 'None'
          in the fib tuple when they are not being used)
"""



"""
1. Import modules and define variables
---------------------------------------
"""

import os

import cfflib

import sys
sys.path.append('/home/jdg45/git_repos/DWI-nipype-interfaces')

import jg_custom_interfaces as jci
import jg_trackvis_interfaces as jti

from nipype.interfaces import io as nio
from nipype.interfaces import utility as util     
from nipype.pipeline import engine as pe

    
# Min number of fibres for a cnxn
n_fib_thresh = 10
n_fib_thresh_list = [1,3,5,8,10]
n_fib_thresh_GROUP = 1
    
# QuickBundles clustering parameters
QB_length_param = 5 
QB_downsample_param = 15
    
QB_length_param_GROUP = 5 
QB_downsample_param_GROUP = 15
    
# Indices of the relevant structures in the .cff file
cnctm_track_num = 0
cnctm_labels_data_num = 3

# Indices of the connectome networks
cnctm_net_num = 0 

flirt_MNI_ref_img = '/usr/local/fsl/x86_64/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

#workflow_base_dir='/work/imaging5/DTI/CBU_DTI/jg_tract_lengths_analysis/\
#Griffiths_Garyfallides_track_normalization_analysis/testing/QB_atlas_analysis'
#workflow_base_dir='/work/imaging5/DTI/CBU_DTI/jg_tract_lengths_analysis/\
#Griffiths_Garyfallides_track_normalization_analysis/testing/QB_atlas_analysis_bayesdirac'
workflow_base_dir='/work/imaging5/DTI/CBU_DTI/jg_tract_lengths_analysis/\
Griffiths_Garyfallides_track_normalization_analysis/testing/QB_atlas_analysis_picodet_twoten'


connectome_data_dir = '/work/imaging5/DTI/CBU_DTI/jg_tract_lengths_analysis/tractography/\
wm_surface_tracks/camino_picodet_twoten/DTI_sequence_64D_1A/\
jg_make_connectomes_from_existing_tracks_workflow/Connectomes'

#connectome_data_dir = '/work/imaging5/DTI/CBU_DTI/jg_tract_lengths_analysis/tractography/\
#wm_surface_tracks/camino_bayesdirac/DTI_sequence_64D_1A/\
#jg_make_connectomes_from_existing_tracks_workflow/Connectomes'


#subject_list = ['CBU100527', 'CBU100631', 'CBU100947', 'CBU100256', 'CBU100948','CBU100562',
#                'CBU100523','CBU100770','CBU100578']
subject_list = ['CBU100160', 'CBU100527']
#subject_list = ['CBU100253']
# Looks like it won't run the 2nd level analysis if you run the 1st level with just a single
# subject because the input ends up being a string (single filename) rather than a list of filenames


"""
2. Specify 1st level workflow
---------------------------------------
"""



# 2a: Define datasource


# 'l1_info' dictionary..
l1_info = dict(cff_file=[['subject_id']],
               bet_file=[['subject_id']], 
               flirt_file=[['subject_id']],
               ROI_file=[['subject_id']])

# Node: 'l1_infosource' - specifies the list of subjects to run the pipeline on
l1_infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                        name="infosource")
l1_infosource.iterables = ('subject_id', subject_list)

# Node:'l1_datasource' - specifies where to grab the input data from
l1_datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                                  outfields=l1_info.keys()),
                        name='l1_datasource')
l1_datasource.inputs.base_directory = connectome_data_dir
l1_datasource.inputs.template = '*'
l1_datasource.inputs.field_template = dict(cff_file='_subject_id_%s/CFFConverter/connectome.cff',
                                           bet_file='_subject_id_%s/bet/data_brain.nii',
                                           flirt_file = '_subject_id_%s/concatxfm/data_brain_flirt_brain_flirt.mat',
                                           ROI_file = '_subject_id_%s/inverse_ROI/ROI_scale500_flirt.nii')
l1_datasource.inputs.template_args = l1_info

# Node: 'l1_inputnode' - Maps the datasource data into the 1st level workflow
l1_inputnode = pe.Node(interface=util.IdentityInterface(fields=['cff_file', 'bet_file', 'flirt_file', 'ROI_file']),
                       name="l1_inputnode")
l1_inputnode.inputs.subjects_dir = connectome_data_dir



# 2b: Apply QuickBundles to connectome files


# Node: 'l1_QB' - First level QuickBundles analysis
l1_QB = pe.Node(interface=jci.apply_QuickBundles_to_connectome_cnxns(), 
              name='l1_QB')   
l1_QB.inputs.QB_downsample_param = QB_downsample_param
l1_QB.inputs.QB_length_param = QB_length_param
l1_QB.inputs.cff_labels_data_number = cnctm_labels_data_num
l1_QB.inputs.cff_network_number = cnctm_net_num # modify
l1_QB.inputs.cff_track_number = cnctm_track_num # modify
#l1_QB.inputs.n_fib_thresh = n_fib_thresh
l1_QB.iterables = ("n_fib_thresh", n_fib_thresh_list)


# 2c: Write out streamlines for each connection to trackvis format (.trk) files
          
    
# Node: #l1_write_QB_Ts_to_trk' - writes l1_QB 'Ts' to .trk files
l1_write_QB_Ts_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_QB_Ts_to_trk')
l1_write_QB_Ts_to_trk.inputs.QB_output_type = 'Ts'
l1_write_QB_Ts_to_trk.inputs.outfile_name_stem = 'QB_Ts_'

# Node: #l1_write_QB_Ets_to_trk' - writes l1_QB 'Ets' to .trk files
l1_write_QB_Ets_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_QB_Ets_to_trk')
l1_write_QB_Ets_to_trk.inputs.QB_output_type = 'Ets'
l1_write_QB_Ets_to_trk.inputs.outfile_name_stem = 'QB_Ets_'

# Node: #l1_write_QB_Vs_to_trk' - writes l1_QB 'Vs' to .trk files
l1_write_QB_Vs_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_QB_Vs_to_trk')
l1_write_QB_Vs_to_trk.inputs.QB_output_type = 'Vs'
l1_write_QB_Vs_to_trk.inputs.outfile_name_stem = 'QB_Vs_'

# Node: #l1_write_QB_Ts_Ets_Vs_to_trk' - writes l1_QB 'Ts', 'Ets', and 'Vs' to single .trk files
l1_write_QB_Ts_Ets_Vs_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_QB_Ts_Ets_Vs_to_trk')
l1_write_QB_Ts_Ets_Vs_to_trk.inputs.QB_output_type = 'Ts_Ets_Vs'
l1_write_QB_Ts_Ets_Vs_to_trk.inputs.outfile_name_stem = 'QB_Ts_Ets_Vs_'


# Node: l1_get_brain_geom - gets volume, # voxels, and diameter values for entire brain
l1_get_brain_geom = pe.Node(interface=jci.read_ROI_geom(),name='l1_get_brain_geom')
l1_get_brain_geom.inputs.use_labels = False

# Node: l1_get_brain_geom - gets volume, # voxels, and diameter values for each ROI
l1_get_ROI_geom = pe.Node(interface=jci.read_ROI_geom(),name='l1_get_ROI_geom')
l1_get_ROI_geom.inputs.use_labels = True
 

# 2d: Apply affine diffusion space --> MNI space transform


# First apply to .pkl outputs of first QB node

# Node: Apply affine transforms to .pkl files - 'Ts', 'Ets', and 'Vs' all together in 'QB.pkl' files 
l1_aff_pkl_QB = pe.Node(interface=jci.apply_flirt_to_fibs(), name='l1_aff_pkl_QB')
l1_aff_pkl_QB.inputs.refspace_image_file = flirt_MNI_ref_img
l1_aff_pkl_QB.inputs.input_type = 'QB_struct'

# Node: Apply affine transforms to .pkl files - 'Ts' 
l1_aff_pkl_Ts = pe.Node(interface=jci.apply_flirt_to_fibs(), name='l1_aff_pkl_Ts')
l1_aff_pkl_Ts.inputs.refspace_image_file = flirt_MNI_ref_img
l1_aff_pkl_Ts.inputs.input_type = 'just_fibs'

# Node: Apply affine transforms to .pkl files - 'Ets' 
l1_aff_pkl_Ets = pe.Node(interface=jci.apply_flirt_to_fibs(), name='l1_aff_pkl_Ets')
l1_aff_pkl_Ets.inputs.refspace_image_file = flirt_MNI_ref_img
l1_aff_pkl_Ets.inputs.input_type = 'just_fibs'

# Node: Apply affine transforms to .pkl files - 'Vs' 
l1_aff_pkl_Vs = pe.Node(interface=jci.apply_flirt_to_fibs(),name='l1_aff_pkl_Vs')
l1_aff_pkl_Vs.inputs.refspace_image_file = flirt_MNI_ref_img
l1_aff_pkl_Vs.inputs.input_type = 'just_fibs'


# Now apply to .trk files (note use of 'mapnode' here for iterating across multiple inputs)

# Node: Apply affine transforms to .trk files - 'Ts' 
l1_aff_trk_Ts = pe.MapNode(interface=jti.trackvis_track_transform(), name='l1_aff_trk_Ts', iterfield=['input_track_file'])
l1_aff_trk_Ts.inputs.reference_volume = flirt_MNI_ref_img    
l1_aff_trk_Ts.inputs.registration_type = 'flirt'

# Node: Apply affine transforms to .trk files - 'Ets' 
l1_aff_trk_Ets = pe.MapNode(interface=jti.trackvis_track_transform(), name='l1_aff_trk_Ets', iterfield=['input_track_file'])
l1_aff_trk_Ets.inputs.reference_volume = flirt_MNI_ref_img    
l1_aff_trk_Ets.inputs.registration_type = 'flirt'

# Node: Apply affine transforms to .trk files - 'Vs'   
l1_aff_trk_Vs = pe.MapNode(interface=jti.trackvis_track_transform(), name='l1_aff_trk_Vs', iterfield=['input_track_file'])
l1_aff_trk_Vs.inputs.reference_volume = flirt_MNI_ref_img    
l1_aff_trk_Vs.inputs.registration_type = 'flirt'



# 2e: Write out the affine-transformed tracks to trackvis format (.trk) files
         

# COMMENTED OUT UNTIL I GET THE AFFINE TRANSFORM ON THE HEADER WORKING     
# Node: #l1_write_aff_pkl_QB_Ts_Ets_Vs_to_trk' - writes l1_aff_pkl_QB 'Ts', 'Ets', 'and 'Vs' to single .trk files
l1_write_aff_QB_Ts_Ets_Vs_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_aff_pkl_QB_Ts_Ets_Vs_to_trk')
l1_write_aff_QB_Ts_Ets_Vs_to_trk.inputs.QB_output_type = 'Ts_Ets_Vs'
l1_write_aff_QB_Ts_Ets_Vs_to_trk.inputs.outfile_name_stem = 'aff_pkl_QB_Ts_Ets_Vs_'

# Node: #l1_write_aff_pkl_QB_Ts_to_trk' - writes l1_aff_pkl_QB 'Ts' to .trk files
l1_write_aff_QB_Ts_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_aff_pkl_QB_Ts_to_trk')
l1_write_aff_QB_Ts_to_trk.inputs.QB_output_type = 'Ts'
l1_write_aff_QB_Ts_to_trk.inputs.outfile_name_stem = 'aff_pkl_QB_Ts_'

# Node: #l1_write_aff_pkl_QB_Ets_to_trk' - writes l1_aff_pkl_QB 'Ets' to .trk files
l1_write_aff_QB_Ets_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_aff_pkl_QB_Ets_to_trk')
l1_write_aff_QB_Ets_to_trk.inputs.QB_output_type = 'Ets'
l1_write_aff_QB_Ets_to_trk.inputs.outfile_name_stem = 'aff_pkl_QB_Ets_'

# Node: #l1_write_aff_pkl_QB_Vs_to_trk' - writes l1_aff_pkl_QB 'Vs' to .trk files
l1_write_aff_QB_Vs_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l1_write_aff_pkl_QB_Vs_to_trk')
l1_write_aff_QB_Vs_to_trk.inputs.QB_output_type = 'Vs'
l1_write_aff_QB_Vs_to_trk.inputs.outfile_name_stem = 'aff_pkl_QB_Vs_'


# 2f: Calculate geometries of ROIs and brain size / volume

# (note: haven't seen these two nodes work in this workflow yet...)

# Node: calculate ROI geometries (volume, # voxels, etc. for each label in the ROI image
l1_calculate_ROI_geoms = pe.Node(interface=jci.read_ROI_geom(), name='l1_calculate_ROI_geoms')
l1_calculate_ROI_geoms.inputs.output_type = 'pickled_dict'
l1_calculate_ROI_geoms.inputs.use_labels = True

# Node: calculate the same geometries from the skull-stripped MRI volume (results are same as using bet brain mask)
l1_calculate_brain_geoms = pe.Node(interface=jci.read_ROI_geom(), name='l1_calculate_brain_geoms')
l1_calculate_brain_geoms.inputs.output_type = 'pickled_dict'
l1_calculate_brain_geoms.inputs.use_labels = False


# 2h: Define workflow


l1_wf = pe.Workflow(name='l1_workflow')
l1_wf.base_dir = workflow_base_dir 

l1_wf.connect([(l1_infosource,l1_datasource,[('subject_id', 'subject_id')])])
l1_wf.connect([(l1_datasource,l1_inputnode, [('cff_file','cff_file'),
                                       ('bet_file','bet_file'),
                                       ('flirt_file','flirt_file'),
                                       ('ROI_file', 'ROI_file')])])


l1_wf.connect([(l1_inputnode,l1_QB,[('cff_file','cff_file')])]) 

l1_wf.connect([(l1_QB,l1_write_QB_Ts_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 
l1_wf.connect([(l1_QB,l1_write_QB_Ets_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 
l1_wf.connect([(l1_QB,l1_write_QB_Vs_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 
l1_wf.connect([(l1_QB,l1_write_QB_Ts_Ets_Vs_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 

l1_wf.connect([(l1_QB,l1_aff_pkl_QB,[('QB_pkl_file_new','in_fibs_file')])]) 
l1_wf.connect([(l1_QB,l1_aff_pkl_Ts,[('Ts_file_new','in_fibs_file')])]) 
l1_wf.connect([(l1_QB,l1_aff_pkl_Ets,[('Ets_file_new','in_fibs_file')])]) 
l1_wf.connect([(l1_QB,l1_aff_pkl_Vs,[('Vs_file_new','in_fibs_file')])]) 

l1_wf.connect([(l1_inputnode,l1_aff_pkl_QB,[('bet_file','DWIspace_image_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Ts,[('bet_file','DWIspace_image_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Ets,[('bet_file','DWIspace_image_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Vs,[('bet_file','DWIspace_image_file')])]) 

l1_wf.connect([(l1_inputnode,l1_aff_pkl_QB,[('flirt_file','flirt_transform_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Ts,[('flirt_file','flirt_transform_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Ets,[('flirt_file','flirt_transform_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_pkl_Vs,[('flirt_file','flirt_transform_file')])]) 

l1_wf.connect([(l1_write_QB_Ts_to_trk,l1_aff_trk_Ts,[('outfile_list','input_track_file')])]) 
l1_wf.connect([(l1_write_QB_Ets_to_trk,l1_aff_trk_Ets,[('outfile_list','input_track_file')])]) 
l1_wf.connect([(l1_write_QB_Vs_to_trk,l1_aff_trk_Vs,[('outfile_list','input_track_file')])]) 

l1_wf.connect([(l1_inputnode,l1_aff_trk_Ts,[('bet_file','source_volume')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_trk_Ets,[('bet_file','source_volume')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_trk_Vs,[('bet_file','source_volume')])]) 

l1_wf.connect([(l1_inputnode,l1_aff_trk_Ts,[('flirt_file','registration_matrix_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_trk_Ets,[('flirt_file','registration_matrix_file')])]) 
l1_wf.connect([(l1_inputnode,l1_aff_trk_Vs,[('flirt_file','registration_matrix_file')])]) 

l1_wf.connect([(l1_aff_pkl_QB, l1_write_aff_QB_Ts_Ets_Vs_to_trk,[('flirted_fibs_file','QB_pkl_file')])]) 
l1_wf.connect([(l1_aff_pkl_QB, l1_write_aff_QB_Ts_to_trk,[('flirted_fibs_file','QB_pkl_file')])]) 
l1_wf.connect([(l1_aff_pkl_QB, l1_write_aff_QB_Ets_to_trk,[('flirted_fibs_file','QB_pkl_file')])]) 
l1_wf.connect([(l1_aff_pkl_QB, l1_write_aff_QB_Vs_to_trk,[('flirted_fibs_file','QB_pkl_file')])]) 

l1_wf.connect([(l1_inputnode,l1_calculate_ROI_geoms,[('ROI_file','ROI_file')])]) 
l1_wf.connect([(l1_inputnode,l1_calculate_brain_geoms,[('bet_file','ROI_file')])]) 


"""
  3. Run 1st level workflow
  --------------------------
"""

if __name__ == '__main__':
    l1_wf.run()
    #l1_wf.write_graph() # NEED TO GET THIS WORKING




"""
4. Specify 2nd level workflow
-----------------------------
"""



# 4a. Define datasource

"""
THIS MORE OR LESS WORKS:
l2_info = dict(l1_aff_pkl_QB_file=[['subject_id']])
l1_subjects_dir = os.path.join(l1_wf.base_dir, l1_wf.name)

# Node:'l2_datasource' - specifies where to grab the outputs of the 1st level analysis
l2_datasource = pe.Node(interface=nio.DataGrabber(),name='l2_datasource')
l2_datasource.inputs.base_directory = workflow_base_dir
#l2_datasource.inputs.template = l1_wf.name + '/_subject_id_*/l1_aff_pkl_QB/flirted_fibs_file.pkl'
#l2_datasource.inputs.template = l1_wf.name + '/_subject_id_*/_n_fib_thresh_1/l1_aff_pkl_QB/flirted_fibs_file.pkl'
template_list = []
[template_list.append(l1_wf.name + '/_subject_id_*/_n_fib_thresh_' + str(t) +
                                    '/l1_aff_pkl_QB/flirted_fibs_file.pkl') for t in n_fib_thresh_list]

l2_datasource.iterables = ('template', template_list)
"""

"""
TRYING THIS OUT FOR BETTER DIRECTORY NAMES:
"""
#l2_info = dict(l1_aff_pkl_QB_file=[['subject_id']])
l1_subjects_dir = os.path.join(l1_wf.base_dir, l1_wf.name)

# Node:'l2_datasource' - specifies where to grab the outputs of the 1st level analysis
l2_datasource = pe.Node(interface=nio.DataGrabber(infields=['n_fib_thresh_list'],outfields=['pkl_file']),name='l2_datasource')
l2_datasource.inputs.base_directory = workflow_base_dir
l2_datasource.inputs.template = '*'
l2_datasource.inputs.field_template = dict(pkl_file=l1_wf.name+'/_subject_id_*/_n_fib_thresh_%s/l1_aff_pkl_QB/flirted_fibs_file.pkl')
l2_datasource.inputs.template_args = dict(pkl_file=[['n_fib_thresh_list']])

l2_datasource.iterables = ('n_fib_thresh_list', n_fib_thresh_list)


# 4b. Run 2nd level QB clustering


# Node: 'l2_QB_Ts' - 2nd level QB analysis on 'Ts' output from 1st level QB 
l2_QB_Ts = pe.Node(interface=jci.apply_QuickBundles_to_QB_pkl_files(),
                   name='l2_QB')    
l2_QB_Ts.inputs.QB_downsample_param = QB_downsample_param_GROUP
l2_QB_Ts.inputs.QB_length_param = QB_length_param_GROUP
l2_QB_Ts.inputs.n_fib_thresh = n_fib_thresh_GROUP
l2_QB_Ts.inputs.fib_type = 'Ts'

# Node: 'l2_QB_Ets' - 2nd level QB analysis on 'Ets' output from 1st level QB 
l2_QB_Ets = l2_QB_Ts.clone(name='l2_QB_Ets')
l2_QB_Ets.inputs.fib_type = 'Ets'

# Node: 'l2_QB_Vs' - 2nd level QB analysis on 'Vs' output from 1st level QB 
l2_QB_Vs = l2_QB_Ts.clone(name='l2_QB_Vs')
l2_QB_Vs.inputs.fib_type = 'Vs'




# 4c. Write out the results in .trk format (this is your atlas!) 


# Node: #l2_write_QB_Ts_to_trk' - writes l2_QB 'Ts' to .trk files
l2_write_QB_Ts_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l2_write_QB_Ts_to_trk')
l2_write_QB_Ts_to_trk.inputs.QB_output_type = 'Ts'
l2_write_QB_Ts_to_trk.inputs.outfile_name_stem = 'l2_write_QB_Ts_to_trk_'

l2_write_QB_Ets_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l2_write_QB_Ets_to_trk')
l2_write_QB_Ets_to_trk.inputs.QB_output_type = 'Ets'
l2_write_QB_Ets_to_trk.inputs.outfile_name_stem = 'l2_write_QB_Ets_to_trk_'

l2_write_QB_Vs_to_trk = pe.Node(interface=jci.write_QuickBundles_to_trk(), name='l2_write_QB_Vs_to_trk')
l2_write_QB_Vs_to_trk.inputs.QB_output_type = 'Vs'
l2_write_QB_Vs_to_trk.inputs.outfile_name_stem = 'l2_write_QB_Vs_to_trk_'



# 4d. Define workflow


l2_wf = pe.Workflow(name='l2_workflow')
l2_wf.base_dir = workflow_base_dir

l2_wf.connect([(l2_datasource,l2_QB_Ts,[('pkl_file','QB_pkl_file_list')])]) 
l2_wf.connect([(l2_datasource,l2_QB_Ets,[('pkl_file','QB_pkl_file_list')])]) 
l2_wf.connect([(l2_datasource,l2_QB_Vs,[('pkl_file','QB_pkl_file_list')])]) 

l2_wf.connect([(l2_QB_Ts,l2_write_QB_Ts_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 
l2_wf.connect([(l2_QB_Ets,l2_write_QB_Ets_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 
l2_wf.connect([(l2_QB_Vs,l2_write_QB_Vs_to_trk,[('QB_pkl_file_new','QB_pkl_file')])]) 


          


"""
5. Run 2nd level workflow
-------------------------
"""

if __name__ == '__main__':
    l2_wf.run()
    #l2_wf.write_graph()



    
