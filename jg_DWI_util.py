"""
-------------------------------------------------------------------------------
 Name:            jg_DWI_util.py
 Purpose:        Collection of python functions for various DWI analyses
 Author:          John
 Created:           ?
 Changelog:
                    - 14 Sep 2011: Added nipype camino tractography function
		    - 07 Nov 2011: Added 'obtain folders from raw data' function
		    - 15 Nov 2011: Added 'analyze_connectome_lengths' function
		    - 23 Nov 2011: Added 'make_CBU_DTI_64D_1A_from_2A'
		    - 24 Nov 2011: Added 'split_multiple_ROI_image_into_integer_ROIs'
		    - 29 Nov 2011: Added 'make_group_connectome_lengths_excel_file'
Docstring:
                    ''jg_DWI_util.py'': code for all DWI analyses
                     (hopefully will prevent from having lots of different code
                     snippets knocking around all the time)

                     Functions:

                                    'run_script'
				    'obtain_folders_from_raw_data'
                                    'normalize_DWIspace_images'
                                    'make_tract_template_fslmaths'
                                    'get_tract_template_Tps'
                                    'write_tract_template_Tps_to_excel_file'
                                    'copy_and_rename_images'
                                    'scale_tractography_by_max_value'
                                    'threshold_tractography_by_numsamples'
                                    'threshold_tractography_by_waytotal'
                                    'probtrackX_tractography'
                                    'camino_tractography'
				    'analyze_connectome_lengths'
				    'make_CBU_DTI_64D_1A_from_2A'
				    'split_multiple_ROI_image_into_integer_ROIs'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import glob
import shutil
import scipy
import numpy as np
import nibabel as nib
import nipype.interfaces.fsl as fsl
import nipype.interfaces.matlab as mlab      # how to run matlab
import nipype.interfaces.spm as spm          # spm
import jg_DWI_util
mlab.MatlabCommand.set_default_matlab_cmd("/usr/local/matlab2008b/bin/glnxa64/MATLAB -nodesktop -nosplash")


def run_script():
	"""
	Edit this section to make scripts to run the functions you want
	Then execute with jg_DWI_util.run_script()
	"""
	import jg_DWI_util
	base_dir = '/work/imaging5/DTI/CBU_DTI/jg_patients_tractography'
	dirs = ['LH_BA44_dorsal', 'LH_BA44_ventral', 'LH_BA45_dorsal', 'LH_BA45_ventral', 'RH_BA44_dorsal', 'RH_BA44_ventral', 'RH_BA45_dorsal', 'RH_BA45_ventral']
	divs = ['divnumsamples', 'divnumsamples_thr0_00001', 'divnumsamples_thr0_00001_mulnumsamples']
	for d in dirs:
		print d
		for ds in divs:
			print ds
			track_dir = os.path.join(base_dir,'subject_data', 'controls', 'all_controls_thresholded_tracks', d, ds)
			images = glob.glob(os.path.join(track_dir, '*'))
			template_name = os.path.join(track_dir, 'track_template_' + d+'_'+ds + '.nii')
			jg_DWI_util.make_tract_template_nibabel(images, template_name)

def obtain_folders_from_raw_data(strings_to_match,strings_to_exclude,raw_data_dir,target_directory, get_MPRAGE):
	"""
	Goes through the raw_data_dir, finds folders that contain the string to match, 
	and (if not present already) copies the to a new folder in the target_directory
	named after the CBUnum
	
	Usage:
	
	jg_DWI_util.obtain_folders_from_raw_data(string_to_match,strings_to_exclude, raw_data_dir,target_directory, get_MPRAGE)
	
	'Strings to match' identifies the folder you want
	'Strings to exclude' allows you to avoid similarly named folders - for example if you want it to find a folder 
	called 'Series_11_CBU_DTI_64D_2A', but ignore 'Series_11_CBU_DTI_64D_2A_FA', then an exclusion string 
	of 'FA' will allow this. I've noticed five potential suffixes like this so far: 'FA', 'MD', 'ADC', 'TRACE', and 'EXP' - 
	so best to include all of these in exclusion strings unless you're after those folders
	
	Inputs:
	
		strings_to_match (e.g. ['CBU_DTI_64D_2A'])
		strings_to_exclude(['FA', 'MD', 'ADC', 'TRACE', 'EXP']
		raw_data_dir (e.g. 'imaging1/raw_data')
		target_directory (currently 'imaging5/DTI/CBU_DTI/raw_data')
		get_MPRAGE = binary (0 or 1) - 1 if you do want MPRAGE to be copied over as well
	
	"""
	for path, subdirs, files in os.walk(raw_data_dir):
		for sd in subdirs:
			keep_dir = 1
			for s2m in strings_to_match:
				if s2m not in sd:
					keep_dir = 0
			for s2e in strings_to_exclude:
				if s2e in sd:
					 keep_dir = 0
			if keep_dir==1:
				fulldir = os.path.join(path,sd)
				CBUnum = 'CBU'+fulldir.split('CBU')[1].split('/')[0].split('_')[0]
				new_subj_dir = os.path.join(target_directory, CBUnum)
				if not os.path.isdir(new_subj_dir):
					print 'making new CBU directory: ' + new_subj_dir
					os.mkdir(new_subj_dir)
					new_data_dir = os.path.join(new_subj_dir, sd)
					print 'shutil copying ' + fulldir + ' to ' + new_data_dir
					shutil.copytree(fulldir, new_data_dir)
					if get_MPRAGE==1:
						MPRAGE_glob = glob.glob(os.path.join(fulldir.split('Series')[0],'*MPRAGE*'))
						if not MPRAGE_glob==[]:
							orig_MPRAGE_dir  = MPRAGE_glob[0]
							print 'orig MPRAGE_dir = ' + orig_MPRAGE_dir
							new_MPRAGE_dir = os.path.join(new_subj_dir, orig_MPRAGE_dir.split('/')[-1])
							print 'new MPRAGE_dir  = ' + new_MPRAGE_dir
							if not os.path.isdir(new_MPRAGE_dir):
								print 'shutil copying ' + orig_MPRAGE_dir + ' to ' + new_MPRAGE_dir
								shutil.copytree(orig_MPRAGE_dir, new_MPRAGE_dir)

def normalize_DWIspace_images(image_files,nodif,T1,seg_sn_file):
	"""
	Add docstring here
	"""	
	DWIspace_images = []
	T1space_images = []
	MNIspace_images = []
	for i in image_files:
		i_filename = i.split('/')[-1]; i_dir = i.split(i_filename)[0]
		if os.path.isfile(i+'.gz'): jg_util.gunzip(i+'.gz')
		if not os.path.isfile(i): print 'image file not present:  '+i
		else: DWIspace_images.append(i), T1space_images.append(os.path.join(i_dir, 'r'+i_filename)), MNIspace_images.append(os.path.join(i_dir,'wr'+i_filename))	
		# Coregister the T1 to the b0 and apply these coregistration parameters to the images
		print 'applying b0-->T1 coregistration parameters to images'
	try:
		jg_coreg = spm.Coregister()
		jg_coreg.inputs.paths = '/usr/local/spm5'
		jg_coreg.inputs.jobtype = 'estwrite'
		jg_coreg.inputs.target = T1
		jg_coreg.inputs.source = nodif
		jg_coreg.inputs.apply_to_files = DWIspace_images
		jg_coreg.run()
	except IOError:	print "problem applying T1-->b0 coregistation parameters to images"
		
	# Apply the normalization parmaeters
	print 'applying normalization parameters to T1 image'
	try:
		jg_norm = spm.Normalize()
		jg_norm.inputs.paths = '/usr/local/spm5'
		jg_norm.inputs.jobtype = 'write'
		jg_norm.inputs.parameter_file = seg_sn_file
		jg_norm.inputs.apply_to_files = T1space_images
		jg_norm.run()
	except IOError:	print "problem with applying normalization parameters"
	print 'done! :)'

def make_tract_template_nibabel(track_images, template_name):
	"""
		Make tract template using purely python (nibabel and numpy) operations
		NOTE: IMAGE LOCATION SEEMS TO BE DIFFERENT TO THAT OUTPUT BY FSLMATHS VERSION OF THIS; 
		I DON'T THINK THE FINAL IMAGES FROM THIS ONE ARE OUTPUT IN THE SAME LOCATION AS THE 
		INPUT TRACK IMAGES. FSLMATHS MAKE TRACK TEMPLATE VERSION IS PREFERRED FOR NOW
		ls
		
		ADD THIS DOC:
			Script for performing the 'tract template analysis' of Hua et al. 2008. 
			'Templates' are probabilistic, stereotaxic maps, where the value at each 
			
			voxel represents the probability that a given tract passes through that 
			voxel. The templates are constructed from constructed by averaging a 
			series of spatially-normalized, binary maps from tractography analyses 
			in a separate group of subjects. The analysis performed here takes scalar 
			maps such as fractional anisotropy (FA) images (spatially normalized to 
			the same space as the templates) and computes a weighted mean across all 
			the voxels within the template, where the weights are provided by the 
			probability values in the template image. Thus, the single number output 
			for each subject/image pair corresponds to the mean voxel intensity within 
			the location of the tract in question, scaled by tract probability, so 
			that voxels where it is highly probable (across the group used to construct 
			the template) that the tract in question passes through them contribute 
			relatively greater to the overall tract probability score than do voxels 
			with less probability that the tract in question passes through them. 
			Usage:
			
		"""
	# 1. Binarize images and sum
	for n in range(0,len(track_images)):
		if n == 0: img_sum = np.float32(nib.load(track_images[0]).get_data()>0)
		else: img_sum = img_sum+np.float32(nib.load(track_images[n]).get_data()>0)
	# 2. Divide by N and write the new template image, getting the affine header from the first image
	nib.Nifti1Image(img_sum/len(track_images), nib.load(track_images[0]).get_affine()).to_filename(template_name)

def make_tract_template_fslmaths(track_images, template_name):
	"""
	Make tract template using fslmaths commands
	NOTE: IMAGE LOCATION SEEMS TO BE DIFFERENT TO THAT OUTPUT BY NIBABEL VERSION OF THIS; 
	FSLMATHS VERSION ONE IS PREFERRED FOR NOW
	"""
	for n in range(0,len(track_images)):
		# 1. Create a set of temporary binarized images and construct an fslmaths command to sum them
		fsl.ImageMaths(in_file = track_images[n], out_data_type = 'float', op_string = '-nan -bin', out_file = 'temp_bin'+str(n)+'.nii.gz').run()
		if n == 0:
			cmd = 'fslmaths ' + 'temp_bin'+str(n)+'.nii.gz'
		else:
			cmd = cmd+' -add ' + 'temp_bin'+str(n)+'.nii.gz'
	# 2. Add 'divide by N' and the new template name, and execute the command
	os.system(cmd+' -div ' + str(len(track_images)) + ' ' + template_name)
	# 3. Remove the temporary images
	os.system('rm temp_bin*')

def get_template_probabilities(scalar_images, template_images):
	"""
	Implements the tract template analysis method of Hua et al. (2008):
	For a list of scalar images, and probabilistic template images,
	calculates a weighted mean of the scalar image voxels, weighted
	by the corresponding template voxels
	Returns an N scalar images x N template images array
	"""
	Tp = np.zeros((len(scalar_images), len(template_images)))  # Note the extra brackets - needed for np.zeros to work (see help)...
	for s in range(0, len(scalar_images)): 
		STp = []
		for t in range(0,len(template_images)):
			# load in the scalar image data, setting nans to zeros
			Si = scipy.nan_to_num(nib.load(scalar_images[s]).get_data())
			Ti = scipy.nan_to_num(nib.load(template_images[t]).get_data())
			# scale the scalar image values by template values
			SiTi = Si*Ti
			# add to output variable
			Tp[s,t] = str(SiTi.sum()/Ti.sum())
	# return the N scalar images x N template images array
	return Tp

def write_tract_template_Tps_to_excel_file(Tps, track_names, sbj_names, xl_filename):
	"""
	Add docstring here
	"""
	from xlwt import Workbook
	wb = Workbook()
	ws = wb.add_sheet('0')
	for t in range(0, len(track_names)):
		for s in range(0, len(sbj_names)):
			if s == 0: ws.write(s,t+1,track_names[t])
			if t == 0: ws.write(s+1,t,sbj_names[s])
			ws.write(s+1,t+1,Tps[s,t])
	wb.save(xl_filename)

def get_percent_lesion_overlap(lesion_template, track_templates):
	"""
	Implements the 'OVERTRACK' tract/lesion overlap method of Thiebaut de Schotten et al. (2008)
	Generates a probabilistc lesion map, then iteratively thresholds at 5%, 10%, 15%...100% of 
	patients, each time overlaying the thresholded lesion map on the binary tract template and 
	calculating the percent overlap
	
	Inputs:
	
	lesion_template = lesion probability map, scaled between 0 and 1
	(use 'make_tract_template_nibabel' function above to generate this from a set of lesion images in standard space)
		
	track_templates = list of BINARY template images to calculate overlap for
	(NOTE: these are different to 'track templates' used in Hua et al.  (2008) type analyses and functions above, in that 
	the track templates here are binary, and not probabilistic)
		
	overlap_image_names = list of names for the overlap images to be generated
		
	
	Outputs:
	
	- overlap_table = list of average % overlap for each track template image. Final column = lesion 
	 probability thresholds (0.1-1), first few columns = ;% overlap at that threshold for each track
	- the corresponding overlap images are written to the filenames specified 
	  in the 'overlap_images' input variable
	
		
	"""
	overlap_vals = [round(o*0.05,2) for o in range(0, 21)]
	overlap_table = np.zeros((len(overlap_vals), len(track_templates)+1)) # = n threshold levels (5%, 10%, 15%, etc.) x n template imgaes
	Li = scipy.nan_to_num(nib.load(lesion_template).get_data())
	for t in range(0,len(track_templates)):
		Ti = scipy.nan_to_num(nib.load(track_templates[t]).get_data())
		for o in range(0, len(overlap_vals)):
			Li_thresh = np.float32(np.greater(Li, overlap_vals[o]))
			Li_thresh_mul_Ti = Li_thresh*Ti
			overlap_table[o,t] = (1/Ti.sum())*Li_thresh_mul_Ti.sum()
			#nib.Nifti1Image(Li_thresh_mul_Ti,
	print overlap_table.shape
	print len(overlap_vals)
	overlap_table[:,overlap_table.shape[1]-1] = overlap_vals 
	return overlap_table


def copy_and_rename_images(orig_images,new_images):
	"""
	Use 'fslmaths mul - 1 ' trick to copy and rename images
	(fsl should then take care of header modifications that 
	simply using linux file renaming ops wouldn't do )
	"""
	for i in range(0, len(orig_images)):
		fsl.ImageMaths(in_file=orig_images[i], out_file=new_images[i],op_string='-mul 1').run()

def scale_tractography_by_max_value(track_image):	
	"""
	Add docstring here
	"""
	fslstats_range = fsl.ImageStats(in_file = track_image, op_string = '-R').run()
	track_image_max = fslstats_range.out_stat[1]
	# (could do this with nibabel instead)
	fsl.ImageMaths(in_file=track_image, out_file = track_image.split('.nii')[0]+'_divmax.nii.gz', op_string = '-div '+str(track_image_max), out_data_type = 'float').run()

def threshold_tractography_by_numsamples(track_image,seed_images, samples_per_voxel, threshold_values):
	"""
	WORKING. (sort out docstring...)
	THRESHOLD VALUES NEEDS TO BE ENTERED AS STRINGS; 'threshold_values = ['0.00001', '0.0001'...]
	**[ MODIFY THIS DOCSTRING = TAKEN FROM ORIGINAL SCRIPT ] **
	Iterates through the control subjects and a subset of the 
	tractography results directories for each subject that constitute
	the final set of results for the present analysis. 
	Notable features:
		- Uses 'os.system(...)' to execute linux command strings
		  (similar to 'unix()' command in matlab)
		- Uses a single line to count the number of nonzero voxels
		  in individual subjects' tractography seed images and 
		  multiply by the number of samples per voxel to get the 
		  total number of samples, using the 'nibabel' python module:
		  nsamp =  nib.load(image).get_data().sum()*5000
		  ...this is considerably simpler and more elegant than 
		  the previous strategy of using 'fslstats -image -V'
		- Uses fslmaths to make renamed copies of images, with 
		  a redundant op string ' -mul 1' that doesn't change anything;
		  this is better than simply using normal file io commands as 
		  these would not make the necessary modifications to the image
		  header
	"""
	numsamples = 0
	for s in seed_images:
		numsamples = numsamples+nib.load(s).get_data().sum()*samples_per_voxel
	print 'dividing by num samples'	
	track_image_divnumsamples = track_image.split('.nii')[0]+'_divnumsamples.nii.gz'
	fsl.ImageMaths(in_file=track_image, op_string = '-div '+str(numsamples), out_file = track_image_divnumsamples).run()
	for t in threshold_values:			
		print '...and thresholding at ' + t
		sig_figs = t.split('.')[-1]
		track_image_divnumsamples_thr = track_image.split('.nii')[0]+'_divnumsamples_thr0_'+sig_figs+'.nii.gz'
		fsl.ImageMaths(in_file = track_image_divnumsamples, op_string = '-thr ' + t, out_file = track_image_divnumsamples_thr).run()

def threshold_tractography_by_waytotal(track_image,waytotal):
	"""
	Need to add this in
	"""
	track_image_divwaytotal = track_image.split('.nii')[0]+'_divwaytotal.nii.gz'
	fsl.ImageMaths(in_file=track_image, op_string = '-div '+str(waytotal), out_file = track_image_divwaytotal).run()
	print '...and thresholding at 0.00001'
	track_image_divwaytotal_thr0_00001 = track_image.split('.nii')[0]+'_divwaytotal_thr0_00001.nii.gz'
	fsl.ImageMaths(in_file=track_image_divwaytotal, op_string = '-thr 0.00001', out_file =track_image_divwaytotal_thr0_00001).run()
	print '...at 0.0001'
	track_image_divwaytotal_thr0_0001 = track_image.split('.nii')[0]+'_divwaytotal_thr0_0001.nii.gz'
	fsl.ImageMaths(in_file=track_image_divwaytotal, op_string = '-thr 0.0001', out_file =track_image_divwaytotal_thr0_0001).run()
	print '...and thresholding at 0.001'
	track_image_divwaytotal_thr0_001 = track_image.split('.nii')[0]+'_divwaytotal_thr0_001.nii.gz'
	fsl.ImageMaths(in_file=track_image_divwaytotal, op_string = '-thr 0.001', out_file =track_image_divwaytotal_thr0_001).run()

def probtrackX_tractography(bedpostX_dir, seeds, waypoints, exclusions, out_dir):
	"""
	Add docstring here
	NOT TESTED YET
	IF THIS ISN'T WORKING TRY CHANGING THE 'MASKS FILE' TO BE WRITTEN IN 
	A DIFFERENT DIRECTORY
	Only set up to work for a 2 seeds + a single waypoint; modify the mask and waypoint file making bits to change this
	Requirements:
		In subject dir:
			- 'tractography/native_space_seeds/from_SPM_inv_norm_params' 
			   folder with seeds in
			- 'nodif_brain_mask' in bedpostX dir must be unzipped
	"""
	if not os.path.isdir(out_dir): os.mkdir(out_dir)
	# make masks file	
	try:
		masks_file = os.path.join(out_dir, 'tractography_masks.txt') # prefixed with 'tractography' to distinguish from FSL-generated files
		masks_file_var = open(masks_file, 'w')
		masks_file_var.write('\n'.join(seeds[0], seeds[1]) + '\n')
		masks_file_var.close()
	except IOError:	print "problem making masks file"
	# make waypoints file
	try:
		waypoints_file = os.path.join(out_dir, 'tractography_waypoints.txt')
		waypoints_file_var = open(waypoints_file, 'w')
		waypoints_file_var.write('\n'.join(waypoints[0], waypoints[1]) + '\n')
		waypoints_file_var.close()
	except IOError:
		print "problem making waypoints file"
	# run tractography
	try:
		jg_probtrackX = fsl.ProbTrackX()
		jg_probtrackX.inputs.seed_file = masks_file	
		jg_probtrackX.inputs.waypoints = waypoints_file
		jg_probtrackX.inputs.avoid_mp = exclusions
		jg_probtrackX.inputs.bpx_directory= bedpostx_dir
		jg_probtrackX.inputs.mask=os.path.join(bedpostX_dir, 'nodif_brain_mask.nii') # this file may need to be unzipped
		jg_probtrackX.inputs.paths_file= os.path.join(out_dir,'fdt_paths')
		jg_probtrackX.inputs.out_dir=out_dir
		jg_probtrackX.inputs.samplesbase_name = os.path.join(bedpostX_dir, 'merged')
		jg_probtrackX.inputs.mode='seedmask' 
		jg_probtrackX.inputs.loop_check=True  
		jg_probtrackX.inputs.c_thresh = 0.1 
		jg_probtrackX.inputs.n_steps=2000 
		jg_probtrackX.inputs.step_length=0.5 
		jg_probtrackX.inputs.n_samples=5000 
		jg_probtrackX.inputs.force_dir=True 
		jg_probtrackX.inputs.opd=True  
		jg_probtrackX.inputs.network = True
		jg_probtrackX.run()
	except IOError:	print "problem running probtrackX"
	"""May need to use the following lines if it doesn't run with masks and waypoints files in the new track dir:
		shutil.copyfile(masks_file, os.path.join(new_track_dir, 'masks.txt'))
		os.remove(masks_file)
		shutil.copyfile(waypoints_file, os.path.join(new_track_dir, 'waypoints.txt'))
		os.remove(waypoints_file)"""

def camino_tractography(subject_dir):
	"""
	Add docstring here
	Mostly stolen from Erik Ziegler's nipype camino+cmtk connectivity tutorial
	[ **RETURN 'workflow' as an output**]
	sort out the 'varargin' style for the input arguments..
	"""
	import nipype.interfaces.io as nio           # Data i/o
	import nipype.interfaces.utility as util     # utility
	import nipype.pipeline.engine as pe          # pypeline engine
	import nipype.interfaces.camino as camino
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.camino2trackvis as cam2trk
	import nipype.algorithms.misc as misc
	import os                                    # system functions
	from jg_DWI_util import get_vox_dims,get_data_dims,get_affine
		
	fsl.FSLCommand.set_default_output_type('NIFTI')
	"""
	Setup for Diffusion Tensor Computation
	--------------------------------------
	In this section we create the nodes necessary for diffusion analysis.
	First, the diffusion image is converted to voxel order.
	"""
	image2voxel = pe.Node(interface=camino.Image2Voxel(), name="image2voxel")
	image2voxel.inputs.in_file = os.path.join(subject_dir, 'data.nii.gz')
	image2voxel.inputs.out_file = os.path.join(subject_dir, 'data.Bfloat')
	image2voxel.run()
		
	fsl2scheme = pe.Node(interface=camino.FSL2Scheme(), name="fsl2scheme")
	fsl2scheme.inputs.usegradmod = True
	fsl2scheme.inputs.bval_file = os.path.join(subject_dir, 'bvals')
	fsl2scheme.inputs.bvec_file = os.path.join(subject_dir, 'bvecs')
	fsl2scheme.inputs.out_file  = os.path.join(subject_dir, 'bvecs.scheme')
	fsfl2scheme.run()
		
	"""
	Second, diffusion tensors are fit to the voxel-order data.
	"""
	dtifit = pe.Node(interface=camino.DTIFit(),name='dtifit')
	dtifit.inputs.in_file = os.path.join(subject_dir, 'data.Bfloat')
	dtifit.inputs.out_file = os.path.join(subject_dir, 'data_DT.Bfloat')
	dtifit.inputs.scheme_file = os.path.join(subject_dir, 'bvecs.scheme')
	dtifit.run()
	"""
	Next, a lookup table is generated from the schemefile and the
	signal-to-noise ratio (SNR) of the unweighted (q=0) data.
	"""
	dtlutgen = pe.Node(interface=camino.DTLUTGen(), name="dtlutgen")
	dtlutgen.inputs.snr = 16.0
	dtlutgen.inputs.inversion = 1
	dtlutgen.inputs.scheme_file = os.path.join(subject_dir, 'bvecs.scheme')
	dtlutgen.inputs.out_file = os.path.join(subject_dir, 'dtLUT')
	"""
	In this tutorial we implement probabilistic tractography using the PICo algorithm.
	PICo tractography requires an estimate of the fibre direction and a model of its 
	uncertainty in each voxel; this is produced using the following node.
	"""
	picopdfs = pe.Node(interface=camino.PicoPDFs(), name="picopdfs")
	picopdfs.inputs.inputmodel = 'dt'
	picopdfs.inputs.luts = os.path.join(subject_dir, 'dtLUT')
	picopdfs.inputs.outputs = data_DT_pdfs.Bdouble
	
	"""
	An FSL BET node creates a brain mask is generated from the diffusion image for seeding the PICo tractography.
	"""
	bet = pe.Node(interface=fsl.BET(), name="bet")
	bet.inputs.mask = True
	bet.inputs.in_file = os.path.join(subject_dir, 'data.nii.gz')
	bet.inputs.out_file = os.path.join(subject_dir, 'data_brain.nii.gz')
	bet.inputs.mask_file = os.path.join(subject_dir, 'data_brain_mask.nii.gz')
	bet.run()
	"""
	Finally, tractography is performed. 
	First DT streamline tractography.
	"""	
	trackdt = pe.Node(interface=camino.TrackDT(), name="trackdt")	
	trackdt.inputs.in_file = os.path.join(subject_dir, 'data_DT.Bfloat')
	trackdt.inputs.seed_file = os.path.join(subject_dir, 'data_brain_mask.nii.gz')
	
	#***UP TO HERE WORKING THROUGH THIS, BUT NOW THE SCRIPT WITH THE PROPER 
	# WORKFLOWS SEEMS TO BE WORKING - SO WILL TRY AGAIN WITH THAT FOR A BIT ****
		
	#trackdt.inputs.out_file = 
	#"""
	#Now camino's Probablistic Index of connectivity algorithm.
	#In this tutorial, we will use only 1 iteration for time-saving purposes.
	#"""	
	#trackpico = pe.Node(interface=camino.TrackPICo(), name="trackpico")
	#trackpico.inputs.iterations = 1
	#""" 
	#**JG: At the moment camino2trackvis isn't working, so remove those parts of the workflow*** 
	#** [ NOTE THE 'min_length' part in the specifications here...need to remove this probably for length analyses]
	#**...the command line call that was generated was:
	#'camino_to_trackvis -i <...trackpico/data_DT_pdfs_tracked> -o data_DT_pdfs_tracked.trk -l 30 -d 96,96,68 -x 2.00000023842,2.00000095367,2.0 --voxel-order LAS'
	#...which gives the error
	#'camino_to_trackvis: 'unrecognized argument: "--voxel-order" '
	#"""	
	##Currently, the best program for visualizing tracts is TrackVis. For this reason, a node is included to
	##convert the raw tract data to .trk format. Solely for testing purposes, another node is added to perform the reverse.
		
	#cam2trk_dt = pe.Node(interface=cam2trk.Camino2Trackvis(), name="cam2trk_dt")
	#cam2trk_dt.inputs.min_length = 30
	#cam2trk_dt.inputs.voxel_order = 'LAS'
		
	#cam2trk_pico = pe.Node(interface=cam2trk.Camino2Trackvis(), name="cam2trk_pico")
	#cam2trk_pico.inputs.min_length = 30
	#cam2trk_pico.inputs.voxel_order = 'LAS'
		
	#trk2camino = pe.Node(interface=cam2trk.Trackvis2Camino(), name="trk2camino")
		
	#"""
	#Tracts can also be converted to VTK and OOGL formats, for use in programs such as GeomView and Paraview,
	#using the following two nodes. For VTK use VtkStreamlines.
	#"""
	#procstreamlines = pe.Node(interface=camino.ProcStreamlines(), name="procstreamlines")
	#procstreamlines.inputs.outputtracts = 'oogl'
	#"""
	#**JG: Adding in a 'vtkstreamlines' node so I can do visualization with paraview
	#(see also the 'c3d' node to convert the structural to the right format )
	#"""
	#vtkstreamlines = pe.Node(interface=camino.VtkStreamlines(),name="vtkstreamlines")
	#"""
	#We can also produce a variety of scalar values from our fitted tensors. The following nodes generate the
	#fractional anisotropy and diffusivity trace maps and their associated headers.
	#"""
	#fa = pe.Node(interface=camino.ComputeFractionalAnisotropy(),name='fa')
	#trace = pe.Node(interface=camino.ComputeTensorTrace(),name='trace')
	#dteig = pe.Node(interface=camino.ComputeEigensystem(), name='dteig')
		
	#analyzeheader_fa = pe.Node(interface= camino.AnalyzeHeader(), name = "analyzeheader_fa")
	#analyzeheader_fa.inputs.datatype = "double"
	#analyzeheader_trace = analyzeheader_fa.clone('analyzeheader_trace')
		
	#fa2nii = pe.Node(interface=misc.CreateNifti(),name='fa2nii')
	#trace2nii = fa2nii.clone("trace2nii")	
	#""" 
	#**JG: Finally: identify the tracks for each region pair
	## CHECK YOUR MATLAB CODE FOR THE REST OF THIS
	#"""
	#procstreamlines_ROIs = pe.Node(interface=camino.ProcStreamlines(), name="procstreamlines_ROIs")	

def analyze_connectome_lengths(cff_file,track_name, endpointsmm_name, labels_name,make_figures,write_text_files,txt_file_out_dir='N/A', txt_file_prefix='N/A'):
	"""
	load connectome file
	Usage: 
	FLs, EDs, FLsubEDs,fib_labels,c_trk_fibres
	fib_lengths, euclidean_distances, fib_lengths_minus_EuDs,fiber_labels,fibre_arrays = jg_DWI_util.analyze_connectome_lengths(cff_file,track_name, endpointsmm_name, labels_name,make_figures,write_text_files,txt_file_out_dir=<txt_file_name>, txt_file_prefix=<txt_file_prefix>):
	"""
	import cfflib
	from jg_DWI_util import scatter_and_hist, scatter_simple
	from nipype.interfaces.cmtk.cmtk import length as fib_length
	import numpy as np
		
	c = cfflib.load(cff_file)		
	# Print summary of connectome file
	print 'printing cff file summary: ' 
	c.print_summary()
		
	# Get fibers as a numpy array
	c_trk = c.get_by_name(track_name)
	c_trk.load()
	c_trk_fibers = c_trk.get_fibers_as_numpy()
		
	# Get corresponding fiber length and endpoint_mm data arrays
	c_endpointsmm = c.get_by_name(endpointsmm_name)
	c_endpointsmm.load()
	EPs = c_endpointsmm.data
		
	c_labels = c.get_by_name(labels_name)
	c_labels.load()
	fib_labels = c_labels.data
		
	# Calculate Euclidean distances
	EDs = []
	for e in range(0,len(EPs)):
		dist = np.sqrt(np.square(EPs[e,0,0]-EPs[e,1,0])+np.square(EPs[e,0,1]-EPs[e,1,1])+np.square(EPs[e,0,2]-EPs[e,1,2]))
		EDs.append(dist)
		
	# Calculate fiber lengths
	FLs = []
	for t in c_trk_fibers:
		FLs.append(fib_length(t))
	# Fiber length minus Euclidean distance:
	FLsubEDs = np.subtract(FLs,EDs) 
	
	## write to text files
	if write_text_files==1:
		np.savetxt(os.path.join(txt_file_out_dir, txt_file_prefix+'_fibre_lengths.txt'),FLs)
		np.savetxt(os.path.join(txt_file_out_dir, txt_file_prefix+'_Euclidean_distances.txt'),EDs)
		np.savetxt(os.path.join(txt_file_out_dir, txt_file_prefix+'_fibre_labels.txt'),fib_labels)
		
	# (write all to a single excel file ? )
	if make_figures == 1:
		# Plot Euclidean distance vs. Track length for all fibers
		x = FLs#FLs[0:10000]
		y = EDs #EDs[0:10000]
		print 'length of x = ' + str(len(x))
		print 'length of y = ' + str(len(y))
		scatter_and_hist(x,y)
		scatter_simple(x,y)
	return FLs, EDs, FLsubEDs,fib_labels,c_trk_fibers

def make_group_connectome_lengths_excel_file(subject_names,cff_files,track_name,endpointsmm_name,labels_name, outfile_name):
	"""
	**NEEDS TESTING**
	write docstring
	"""
	import jg_DWI_util
	from xlwt import Workbook
	
	wb = Workbook()
	ws = wb.add_sheet('0')
	col_ind = 1
	for c in range(0, len(cff_files)):
		print 'processing subject ' + str(subject_names[c])
		flen, eud, fl = jg_DWI_util.analyze_connectome_lengths(cff_files[c], track_name,endpointsmm_name,labels_name,0,0)
		col_ind = col_ind+6
		ws.write(0,col_ind,subject_names[c])
		# for each subject, set out a block in the spreadsheet and write in the three variables
		for f in range(0, len(flen)):
			ws.write(f+1,col_ind+1,str(flen[f]))
			ws.write(f+1,col_ind+2,str(eud[f]))
			ws.write(f+1,col_ind+3,str(fl[f][0]))
			ws.write(f+1,col_ind+4,str(fl[f][1]))
		# (also get the mean, stdev, etc. fibre stats and put in on an edgewise,rather than fibrewise, basis)
	# write the excel file
	wb.save(outfile_name)	

def fosvtk_show_fibres_with_labels(fibres, labels):
	"""
	STILL IN DEVELOPMENT
	"""
	#from dipy.tracking import metrics as tm
	#from dipy.trackivs import distances as td
	from dipy.viz import fvtk
	#from nibabel import trackvis as tv
	
	## load trackvis streams
	#streams,hdr=tv.read(trk_name)
	## copy tracks
	## downsample - will avoid that
	r = fvtk.ren()
	## 'colors' is an array of numbers the same size as the number of trackvs
	fvtk.add(r,fvtk.line(fibres, labels, opacity=1))
	#fvtk.record(r.n_frames=1,out_path='/home/jdg45/fvtk_mov', size=(600,600))
	
	#fvtk.clear(r)
	
def rewrite_trk_file_with_ED_vs_FL_scalars(trk_file_orig,trk_file_new, scalar_type):
	"""
	Read in a trackvis file, calculate the Euclidean distance between
	the start and endpoints of each fibre, and write out a new trackvis
	file where each streamline is ooloured according to length, length-ED, 
	or %ED of L
	"""	
	import nibabel as nib
	import numpy as np
	from nipype.interfaces.cmtk.cmtk import length as fib_length
	fibres_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
	hdr_new = hdr_orig.copy()
	outstreams = []
	for f in fibres_orig:
		# Calculate fiber lengths	
		FL = fib_length(f[0]) 
		# Calculate Euclidean distance between fibre start and endpoints
		ED = np.sqrt(np.square(f[0][0][0]-f[0][-1][0])+np.square(f[0][0][1]-f[0][-1][1])+np.square(f[0][0][2]-f[0][-1][2]))
		# Fiber length minus Euclidean distance:
		FLsubED = np.subtract(FL, ED)
		ED_as_percent_of_FL = np.divide(100,FL)*ED
		if scalar_type == 'FL':
			scalar_array = np.ones((len(f[0]),1),dtype='float')*FL
			property_array = np.array([FL], dtype='float32')
		if scalar_type == 'ED':
			scalar_array = np.ones((len(f[0]),1),dtype='float')*ED
			property_array = np.array([ED], dtype='float32')
		if scalar_type == 'FLsubED':
			scalar_array = np.ones((len(f[0]),1),dtype='float')*FLsubED
			property_array = np.array([FLsubED], dtype='float32')
		if scalar_type == 'ED_as_percent_of_FL':
			scalar_array = np.ones((len(f[0]),1),dtype='float')*ED_as_percent_of_FL
			property_array = np.array([ED_as_percent_of_FL], dtype='float32')
		new_tuple=tuple([f[0], scalar_array,property_array])				
		outstreams.append(new_tuple)
	n_fib_out = len(outstreams)
	hdr_new['n_count'] = n_fib_out	
	hdr_new['n_scalars'] = np.array(1, dtype='int16')				#hdr_new['scalar_name'] = np.array(['JG_COLOURS', '', '', '', '', '', '', '', '', ''],dtype='|S20')		
	hdr_new['scalar_name'] = np.array([scalar_type, '', '', '', '', '', '', '', '', ''],dtype='|S20')
	hdr_new['n_properties'] = np.array(1, dtype='int16')
#	hdr_new['property_name'] = np.array(['JG_PROPERTY', '', '', '', '', '', '', '', '', ''],dtype='|S20')
	hdr_new['property_name'] = np.array([scalar_type, '', '', '', '', '', '', '', '', ''],dtype='|S20')
	nib.trackvis.write(trk_file_new, outstreams, hdr_new)

def make_trk_file_for_2_connectome_nodes(cff_file,node_indices,trk_file_orig, trk_file_new, colour_array = None):
	"""
	takes a .cff connectome file, a
	corresponding .trk trackvis file, 
	and a pair of node indices. If there are 
	any fibres connecting those two regions, 
	outputs them in a new trackvis file
	
	If a colour array is passed, the fibres
	coloured (with the scalars field) according
	to the values in that array
		 
	"""
	import cfflib
	import nibabel as nib
	c = cfflib.load(cff_file)
	c_labels = c.get_connectome_data()[3]
	c_labels.load()
	#c_fibres = c.get_by_name('Tract file 0').load()
	#c_fibres_array = c_fibres. 
	fibres_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
	
	node_indices_reversed = []
	node_indices_reversed.append(node_indices[1])
	node_indices_reversed.append(node_indices[0])
	
	track_indices = []
	for i in range(0, len(c_labels.data)):		
		if c_labels.data[i][0] == node_indices[0]:
			if c_labels.data[i][1] == node_indices[1]:
				track_indices.append(i)
		if c_labels.data[i][0] == node_indices[1]:
			if c_labels.data[i][1] == node_indices[0]:
				track_indices.append(i)
			
	if not track_indices == []:	
		hdr_new = hdr_orig.copy()
		outstreams = []
		for i in track_indices:
			if not colour_array == None:
				# trying properties briefly; scalars doesn't read properly atm
#				scalar_array = np.ones((len(fibres_orig[i][0]),1),dtype='float32')*colour_array[i]
				scalar_array = np.ones((len(fibres_orig[i][0]),1),dtype='float')*colour_array[i]
				property_array = np.array([colour_array[i]], dtype='float32')
				#new_tuple = tuple([fibres_orig[i][0], scalar_array, None])
				#new_tuple=tuple([fibres_orig[i][0], scalar_array,property_array])				
				new_tuple=tuple([fibres_orig[i][0], scalar_array,property_array])
				hdr_new['n_scalars'] = np.array(1, dtype='int16')
				hdr_new['scalar_name'] = np.array(['JG_COLOURS', '', '', '', '', '', '', '', '', ''],dtype='|S20')		
				hdr_new['n_properties'] = np.array(1, dtype='int16')
				hdr_new['property_name'] = np.array(['JG_PROPERTY', '', '', '', '', '', '', '', '', ''],dtype='|S20')				
				outstreams.append(new_tuple)
			else:
				outstreams.append(fibres_orig[i])
		n_fib_out = len(outstreams)
		hdr_new['n_count'] = n_fib_out	
		nib.trackvis.write(trk_file_new, outstreams, hdr_new)	
	else: 
		print ' no tracks found for ROIs ' + str(node_indices[0]) + ' and ' + str(node_indices[1])
	#return track_indices
	
	
def read_Lausanne2008_ROI_list(ROI_list_xl_file):
	"""
	**Add documentation**
	Note: assumes that the ROI list starts from 
	the 3rd row in the excel file (first two are
	headers and are ignored)
	The first column in the excel document contains
	the ROI numbers, the second contains the region
	names
	"""
	import xlrd
	newdict = {}
	wb = xlrd.open_workbook(ROI_list_xl_file)
	sh = wb.sheets()[0]
	for r in range(2, len(sh.col_values(0))):
		newdict[int(sh.col_values(0)[r])] = sh.col_values(1)[r] 
	return newdict		

def make_trk_file_for_many_connectome_nodes_from_ROI_list(ROI_list_xl_files,cff_file,trk_file_orig, trk_file_new_base, n_fib_thresh=0):
	"""
	Generalization of 'make_trk_file_for_2_connectome_nodes' to 2 sets of nodes. 
	3 Main differences:
	
	1. Node pairs are read from rows in excel file(s), rather than an indices
	2. Only a base outputfile stem, rather than the full output filename,
	   needs to be provided, as because multiple output files are produced
	3. No option to colour fibres with scalars in this function
	
	In order to find connections within one set of nodes, just supply one excel file, 
	in which case the function will only check half the connections
	
	"""	
	import cfflib
	import nibabel as nib
	c = cfflib.load(cff_file)
	c_labels = c.get_connectome_data()[3]
	c_labels.load()
	c_net = c.get_connectome_network()[0]
	c_net.load()
	#c_fibres = c.get_by_name('Tract file 0').load()
	#c_fibres_array = c_fibres. 
	fibres_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
	if len(ROI_list_xl_files)==2:
		two_node_lists = 1
		ROI_list_dict1 = jg_DWI_util.read_Lausanne2008_ROI_list(ROI_list_xl_files[0])
		ROI_list_dict2 = jg_DWI_util.read_Lausanne2008_ROI_list(ROI_list_xl_files[1])
	else:
		two_node_lists = 0
		ROI_list_dict1 = jg_DWI_util.read_Lausanne2008_ROI_list(ROI_list_xl_files)
		ROI_list_dict2 = ROI_list_dict1	
	for k in range(0, len(ROI_list_dict1.keys())):
		for kk in range(0,len(ROI_list_dict2.keys())):
			ROI1_name = str(ROI_list_dict1.values()[k])
			ROI1_number = int(ROI_list_dict1.keys()[k])
			ROI2_name = str(ROI_list_dict2.values()[kk])
			ROI2_number = int(ROI_list_dict2.keys()[kk])
			trk_file_new = trk_file_new_base+'_'+str(ROI1_number)+'_'+ROI1_name+'__to__'+str(ROI2_number)+'_'+ROI2_name+'.trk'
			node_indices = [ROI1_number,ROI2_number]
			node_indices_reversed = [ROI2_number, ROI1_number]
			track_indices = []
			a = np.nonzero(c_labels.data==ROI1_number)[0]
			#print 'a = ' + str(a)
			b = np.nonzero(c_labels.data==ROI2_number)[0]
			#print 'b = ' + str(b)
			if ROI1_number in c_net.data.edge[ROI2_number]:
				n_fibs = c_net.data.edge[ROI2_number][ROI1_number]['number_of_fibers']
			elif ROI2_number in c_net.data.edge[ROI1_number]:
				n_fibs = c_net.data.edge[ROI1_number][ROI2_number]['number_of_fibers']
			else: n_fibs = 0
			if n_fibs>=n_fib_thresh:
				#print 'node indices = ' + str(ROI1_number) + ' ' + str(ROI2_number)
				for a_int in a:
					if a_int in b:
						if two_node_lists == 0:
							if kk>k:
								track_indices.append(a_int)
								print 'found track - index ' + str(a_int) + ' , ROIs ' + str(ROI1_number) + ', ' + str(ROI2_number)	
						else: 
							track_indices.append(a_int)
							print 'found track - index ' + str(a_int) + ' , ROIs ' + str(ROI1_number) + ', ' + str(ROI2_number)
			if not track_indices == []:	
				hdr_new = hdr_orig.copy()
				outstreams = []
				for i in track_indices:
					outstreams.append(fibres_orig[i])
				n_fib_out = len(outstreams)
				hdr_new['n_count'] = n_fib_out	
				nib.trackvis.write(trk_file_new, outstreams, hdr_new)	
				#else: 
				#	print ' no tracks found for ROIs ' + str(node_indices[0]) + ' and ' + str(node_indices[1])
				#return track_indices
	

def combine_connectome_node_volumes(ROI_list_xl_file,ROI_img, new_file):
	"""
	Combines all regions listed in the excel file into a single image volume,
	retaining the numbering. Or, can be thought of as sub-sampling the ROI_image_file
	by removing voxels whose intensity value is not listed in the ROI_list_xl_file
	"""
	import os
	from nipype.interfaces import fsl
	fsl.FSLCommand.set_default_output_type('NIFTI')
	temp_img = new_file.split('.nii')[0]+'_temp.nii'
	ROI_list_dict = jg_DWI_util.read_Lausanne2008_ROI_list(ROI_list_xl_file)
	thresh_val = str(ROI_list_dict.keys()[0])
	fsl.ImageMaths(in_file=ROI_img, op_string=' -thr ' +thresh_val + ' -uthr ' + thresh_val,
		       out_file=new_file).run()
	for k in range(1, len(ROI_list_dict.keys())):
		thresh_val = str(ROI_list_dict.keys()[k])
		print 'node ' + thresh_val
		fsl.ImageMaths(in_file=ROI_img, op_string=' -thr ' +thresh_val + ' -uthr ' + thresh_val,
		out_file=temp_img).run()
		fsl.ImageMaths(in_file=temp_img,in_file2=new_file, op_string = ' -add ', out_file=new_file).run()
	os.remove(temp_img)
		
		
	
def scatter_and_hist(x,y):
	#( from http://matplotlib.sourceforge.net/examples/pylab_examples/scatter_hist.html )
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.ticker import NullFormatter
		
	nullfmt = NullFormatter()	
	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left+width+0.02
		
	rect_scatter=[left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]
		
	# Start with a rectangular figure
	plt.figure(1, figsize=(8,8))
		
	axScatter=plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)
		
	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)
		
	# the scatter plot
	axScatter.scatter(x,y)
		
	# now determine nice limis by hand
	#binwidth=0.25
	#xymax=np.max([np.max(np.fabs(x),np.max(np.fabs(y)))])
		
		
	#lim=(int(xymax/binwidth)+1)*binwidth
		
	#axScatter.set_xlim( (-lim,lim) )
	#axScatter.set_ylim( (-lim,lim) )
		
	#bins = np.arange(-lim,lim+binwidth,binwidth)
	#axHistx.hist(x,bins=bins)
	#axHisty.hist(y,bins=bins, orientation= 'horizontal')
		
	#axHistx.set_xlim( axScatter.get_xlim() )
	#axHisty.set_ylim( axScatter.get_ylim() )
		
	plt.show()
	
def scatter_simple(x,y):
	"""
	stolen from matplotlib examples
	"""
	import pylab
	N = len(x)
	area = pylab.pi*(10*pylab.rand(N))**2 # 0 to 10 point radiuses
	pylab.scatter(x,y,s=area,marker='^', c='r')
	pylab.show()

def make_CBU_DTI_64D_1A_from_2A(orig_image_file,new_image_file,orig_bvals_file,new_bvals_file,orig_bvecs_file,new_bvecs_file):
	"""
	Uses numpy to take the first 65 bvecs and bvals, and uses the nipype interface 'fsl.ExtractROI'
	(which wraps the fsl program 'fslsplit') to extract the first 65 volumes of the image
	
	Usage:
			jg_DWI_util.make_CBU_DTI_64D_1A_from_2A(orig_image_file,new_image_file,orig_bvals_file,new_bvals_file,orig_bvecs_file,new_bvecs_file)
			
	Inputs:
			orig_image_file = filename of original data_raw.nii.gz 4D diffusion weighted image volume
			new_image_file = filename of new data image
			orig_bvals_file = filename of original bvals file (e.g. from cbu_write_fdt)
			new_bvals_file = filename of new bvals file
			orig_bvecs_file = filename of original bvecs file (e.g. from cbu_write_fdt)
			new_bvecs_file = filename of new bvecs file
	"""
	import numpy as np
	from nipype.interfaces import fsl	
	print 'extracting first 65 bvals and bvecs'
	orig_bvals = np.loadtxt(orig_bvals_file)
	new_bvals = orig_bvals[:][0:64]
	orig_bvecs = np.loadtxt(orig_bvecs_file)
	np.savetxt(new_bvals_file,new_bvals)
	new_bvecs = orig_bvecs[:][0:64]
	np.savetxt(new_bvecs_file,new_bvecs)
	print 'extracting first 65 volumes'
	split = fsl.ExtractROI(in_file=orig_image_file, roi_file = new_image_file, t_min = 0, t_size = 65)
	print 'command = ' + split.cmdline
	split.run()
	
def split_multiple_ROI_image_into_integer_ROIs(multiple_ROI_image, integer_ROI_image_base_name, integers_to_extract):
	"""
	NOT TESTED YET. 
	"""
	for intgr in integers_to_extract:
		# Take 0.1 either side of the integer value the avoid any numerical innaccuracies
		lowerthresh= str(intgr-0.1)
		upperthresh=str(intgr+0.1)
		new_image=integer_ROI_image_base_name+'_ROI'+str(intgr)+'.nii.gz'
		fim = fsl.ImageMaths(in_file=multiple_ROI_image, out_file=new_image,op_string= ' -thr ' +lowerthresh+ ' -uthr ' + upperthresh)
		print 'running ' + fim.cmdline
		fim.run()



"""
The following five functions ('get_vox_dims', 'get_data_dims', 'get_affine', 'select_aparc', and 'select_aparc_annot'
are taken from and used by Erik Ziegler's nipype camino tutorial, as well as my re-hash of it above.

Here's the spiel for them from the tutorial code:

"We define the following functions to scrape the voxel and data dimensions of the input images. This allows the
pipeline to be flexible enough to accept and process images of varying size. The SPM Face tutorial
(spm_face_tutorial.py) also implements this inferral of voxel size from the data. We also define functions to
select the proper parcellation/segregation file from Freesurfer's output for each subject. For the mapping in
this tutorial, we use the aparc+seg.mgz file. While it is possible to change this to use the regions defined in
aparc.a2009s+aseg.mgz, one would also have to write/obtain a network resolution map defining the nodes based on those
regions."

"""
def get_vox_dims(volume):
	"""
	(Stolen from Erik Ziegler's nipype camino tutorial; see above)
	"""
	import nibabel as nb
	if isinstance(volume, list):
		volume = volume[0]
	nii = nb.load(volume)
	hdr = nii.get_header()
	voxdims = hdr.get_zooms()
	return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

def get_data_dims(volume):
	"""
	(Stolen from Erik Ziegler's nipype camino tutorial; see above)
	"""
	import nibabel as nb
	if isinstance(volume, list):
		volume = volume[0]
	nii = nb.load(volume)
	hdr = nii.get_header()
	datadims = hdr.get_data_shape()
	return [int(datadims[0]), int(datadims[1]), int(datadims[2])]

def get_affine(volume):
	"""
	(Stolen from Erik Ziegler's nipype camino tutorial; see above)
	"""
	import nibabel as nb
	nii = nb.load(volume)
	return nii.get_affine()

def select_aparc(list_of_files):
	"""
	(Stolen from Erik Ziegler's nipype camino tutorial; see above)
	"""
	for in_file in list_of_files:
		if 'aparc+aseg.mgz' in in_file:
			idx = list_of_files.index(in_file)
	return list_of_files[idx]

def select_aparc_annot(list_of_files):
	"""
	(Stolen from Erik Ziegler's nipype camino tutorial; see above)
	"""
	for in_file in list_of_files:
		if '.aparc.annot' in in_file:
			idx = list_of_files.index(in_file)
	return list_of_files[idx]

