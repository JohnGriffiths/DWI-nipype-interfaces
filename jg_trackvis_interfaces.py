"""
================================================================================
 jg_trackvis_interfaces.py - nipype wrappers for trackvis command line functions
================================================================================

                                                         JDG 11/01/2012  
    
    (NOTE: 'beta' version; not necessarily fully functioning atm)


Contents:
---------

	- trackvis_track_transform
	
"""

class trackvis_track_transform_InputSpec(CommandLineInputSpec):

	input_track_file = File(exists=True, argstr='%s',desc='name of input .trk file', position=1)
	
	output_track_file = traits.String(argstr='%s',genfile=True,Mandatory=False,
									  desc='name of output .trk file', position=2)	
	
	source_volume = File(exists=True, argstr='-src %s', mandatory=True,
        				 desc='track source volume filename')
	
	reference_volume = File(exists=True, argstr='-ref %s', mandatory=True,
							desc='track source volume filename')
	
	registration_matrix_file= File(exists=True, argstr='-reg %s', mandatory=True,
								   desc='registration matrix file')	
	
	registration_type = traits.Enum('flirt', 'tkregister', argstr='-reg_type %s',
								     desc='registration type - ''flirt'' or ''tkregister''') 
	
	invert_reg = traits.Bool(argstr=' -invert_reg ', desc="invert reg")

class trackvis_track_transform_OutputSpec(TraitedSpec):
	
	output_track_file = File(exists=True, desc=' Track transform output file ')	

class trackvis_track_transform(CommandLine):
	"""	
	Trackvis instructions:
	
		Usage: track_transform INPUT_TRACK_FILE OUTPUT_TRACK_FILE [OPTION]...
		
		  -src, --source_volume <filename> 
		           source volume file that the original tracks are based on, usually 
		           dwi or b0 volume. must be in nifti format.
		  -ref, --reference_volume <filename> 
		           reference volume file that the tracks are registered to.
		           must be in nifti format.
		  -reg, --registration_matrix_file <filename>
		           registration matrix file from source to reference volume.
		  -invert_reg, --invert_registration_matrix
		           invert the registration matrix. for convenience of inverse 
		           transformation. 
		  -reg_type, --registration_type <type>
		           type of the registration matrix. valid inputs are 'flirt' or
		           'tkregister'. default is 'flirt'.
		  -h, --help
		           display this help
		 		
		Example: 
		
		  track_transform tracks.trk tracks_new.trk -src dti_b0.nii -ref brain.nii -reg reg.mtx

	"""
	_cmd = 'track_transform '
	input_spec=trackvis_track_transform_InputSpec
	output_spec=trackvis_track_transform_OutputSpec

	def _list_outputs(self):
		outputs = self.output_spec().get()
		outputs['output_track_file'] = os.path.abspath(self._gen_outfilename())
		return outputs

	def _gen_filename(self, name):
		if name is 'output_track_file':
			return self._gen_outfilename()
		else:
			return None
	def _gen_outfilename(self):
		_, name , _ = split_filename(self.inputs.input_track_file)
		return name + '_tt.trk'


