from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec, File
from nipype.utils.filemanip import split_filename

import os
import numpy as np
import nibabel as nib
from nipype.interfaces.cmtk.cmtk import length as fib_length

class rewrite_trk_file_with_ED_vs_FL_scalars_InputSpec(BaseInterfaceInputSpec):
	
	trk_file_orig = File(exists=True, desc='original track file', mandatory=True)
	trk_file_new = traits.String(argstr = '%', desc='name of new track file to be made', mandatory=True)
	scalar_type = traits.String(argstr='%s',desc='Type of scalar...', mandatory=True)
	
class rewrite_trk_file_with_ED_vs_FL_scalars_OutputSpec(TraitedSpec):
	
	trk_file_new = File(exists=True, desc="trk_file_new")
	
class rewrite_trk_file_with_ED_vs_FL_scalars(BaseInterface):
	"""
	Reads in a trackvis file and writes out a copy of the file
	with scalars and properties added to each fibre (streamline), according
	to one of four options, related to the length of the streamline:
		
		1. Fibre length  ('FL')
		2. Euclidean distance between endpoints ('ED')
		3. Difference between FL and ED ('FL_sub_ED')
		4. ED as a percentage of FL ('ED_pco_FL')
		
	Each of these is a single number for each fibre. This number
	is written into both the 'properties' and 'scalars' fields of the 
	trackvis file format. For the scalars (which is a 1xN array for each
	3xN streamline array), the same number is written at every point 
	along the fibre. 
	
	Usage: 
		rewrite_trk_file_with_ED_vs_FL_scalars(trk_file_orig, trk_file_new, scalar_type)
	 
	Inputs:
	
		trk_file_orig - original .trk file
		trk_file_new  - name of new .trk file to be written 
		scalar_type   - type of scalar to write ('FL', 'ED', 'FL_sub_ED', 'ED_pco_FL'; see above)
	"""
	input_spec = rewrite_trk_file_with_ED_vs_FL_scalars_InputSpec
	output_spec = rewrite_trk_file_with_ED_vs_FL_scalars_OutputSpec
	def _run_interface(self,runtime):
		scalar_type = self.inputs.scalar_type
		trk_file_orig = self.inputs.trk_file_orig
		trk_file_new = self.inputs.trk_file_new
		fib_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
		hdr_new = hdr_orig.copy()
		fib_new = []
		for f in fib_orig:
			# Calculate fibre lengths	
			FL = fib_length(f[0]) 
			# Calculate Euclidean distance between fibre start and endpoints
			ED = np.sqrt(np.square(f[0][0][0]-f[0][-1][0])+np.square(f[0][0][1]-f[0][-1][1])+np.square(f[0][0][2]-f[0][-1][2]))
			# Calculate fibre length minus Euclidean distance:
			FL_sub_ED = np.subtract(FL, ED)
			# Calculate Euclidean distance as a percentage of fibre length
			ED_pco_FL = np.divide(100,FL)*ED
			if scalar_type == 'FL':
				scalar_array = np.ones((len(f[0]),1),dtype='float')*FL
				property_array = np.array([FL], dtype='float32')
			if scalar_type == 'ED':
				scalar_array = np.ones((len(f[0]),1),dtype='float')*ED
				property_array = np.array([ED], dtype='float32')
			if scalar_type == 'FL_sub_ED':
				scalar_array = np.ones((len(f[0]),1),dtype='float')*FL_sub_ED
				property_array = np.array([FL_sub_ED], dtype='float32')
			if scalar_type == 'ED_pco_FL':
				scalar_array = np.ones((len(f[0]),1),dtype='float')*ED_pco_FL
				property_array = np.array([ED_pco_FL], dtype='float32')
			new_tuple=tuple([f[0], scalar_array,property_array])				
			fib_new.append(new_tuple)
		n_fib_out = len(fib_new)
		hdr_new['n_count'] = n_fib_out	
		hdr_new['n_scalars'] = np.array(1, dtype='int16')
		hdr_new['scalar_name'] = np.array([scalar_type, '', '', '', '', '', '', '', '', ''],dtype='|S20')
		hdr_new['n_properties'] = np.array(1, dtype='int16')
		hdr_new['property_name'] = np.array([scalar_type, '', '', '', '', '', '', '', '', ''],dtype='|S20')
		nib.trackvis.write(trk_file_new, fib_new, hdr_new)		
		return runtime
	
	def _list_outputs(self):
		outputs = self._outputs().get()
		fname = self.inputs.trk_file_new
		outputs["trk_file_new"] = fname
		return outputs
	

