
"""
***************************************************

Interfaces for various Track length etc. functionalities:

	- rewrite_trk_file_with_ED_vs_FL_scalars
	- make_trk_files_for_connectome_node_list
	- read_ROI_list
			
***************************************************
"""

import os
import numpy as np
import nibabel as nib
from nipype.interfaces.cmtk.cmtk import length as fib_length
import cfflib


from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits,
                                    TraitedSpec, File, StdOutCommandLine,
                                    StdOutCommandLineInputSpec, BaseInterface, BaseInterfaceInputSpec, isdefined)
from nipype.utils.filemanip import split_filename




class rewrite_trk_file_with_ED_vs_FL_scalars_InputSpec(BaseInterfaceInputSpec):
	
	trk_file_orig = File(exists=True, desc='original track file', mandatory=True)
	trk_file_new = traits.String(argstr = '%s', desc='name of new track file to be made', mandatory=True)
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
		r = jg_nipype_interfaces.rewrite_trk_file_with_ED_vs_FL_scalars()
		r.inputs.trk_file_orig = <trk_file_orig>
		r.inputs.trk_file_new  = <trk_file_new>
		r.inputs.scalar_type   = <scalar_type>
		r.run()
	 
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
		print 'loading fibres...'
		fib_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
		print str(len(fib_orig)) + ' fibres loaded'
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
	


class make_trk_files_for_connectome_node_list_InputSpec(BaseInterfaceInputSpec):
	
	ROI_xl_file1 = File(exists=True, desc='excel file with list of node ROI numbers to identify', mandatory=True)
	ROI_xl_file2 = File(exists=True, desc='second excel file with list of node ROI numbers to identify', mandatory=False)
	cff_file = File(exists=True, desc='.cff (connectome file format file)', mandatory=True)
	trk_file_orig = File(exists=True, desc='original track file', mandatory=True)
	trk_file_new = traits.String(argstr = '%s', desc='base name of name of new track files to be made', mandatory=True)
	n_fib_thresh = traits.Int(1,desc='minimum number of fibres between node pairs', usedefault=True)
	cff_track_name =traits.String('Tract file 0', desc='name of track file in cff object. Default = ''Tract file 0', usedefault=True)
	cff_labels_name =traits.String('data_DT_pdfs_tracked_filtered_fiberslabel', desc='name of labels array in cff object. Default = ' 'data_DT_pdfs_tracked_filtered_fiberslabel',usedefault=True)
	cff_network_number = traits.Int(0,desc='number (index - e.g. 0, 1, 2) of network file in cff object. Default is 0', usedefault=True)	
	# (not using a name and the '.get_by_name' cff method for this atm because there isn't a standard default name for the network (Swederik's
	#  workflows use subject number, which is not constant across subjs), and I'm not sure hwo to tell nipype to do something when nothing has
	# been entered for non-mandatory inputs but there is no default value (the default would be to use the network index number, as is currently
	# being used). I think it needs to be something like 'if self.inputs.cff_network_name==None: ', but 'None' doesn't work	
	

class make_trk_files_for_connectome_node_list_OutputSpec(TraitedSpec):
	trk_file_new = File(exists=False, desc="trk_file_new")
	
	
	
class make_trk_files_for_connectome_node_list(BaseInterface):
	"""
	Outputs a new .trk file containing a subset of the fibres in the 
	trk_file_orig input, selected according to either one or two excel files,
	each with 2 columns - one for node numbers and one for region names.
	Input a single excel file for for cnxns within one group of nodes, or 
	two excel files for cnxns between two groups of nodes. A number of fibres
	threshold can be used to exclude 'low probability' connections.
	
	Usage: 
		m = jg_nipype_interfaces.make_trk_files_for_connectome_node_list()
		m.inputs.trk_file_orig =  <original track file>
		m.inputs.trk_file_new  =  <new track file to be written>
		m.inputs.ROI_xl_file1  =  <excel list of ROI numbers
		m.inputs.ROI_xl_file2  =  (leave blank if only looking@cnxns within 1 group of nodes)
		m.inputs.cff_file      =  <connectome file>
		m.inputs.n_fib_thresh  =  <minimum number of fibres for each cnxn>
		m.run()
		
		
	"""
	input_spec = make_trk_files_for_connectome_node_list_InputSpec
	output_spec = make_trk_files_for_connectome_node_list_OutputSpec
	
	def _run_interface(self,runtime):
		print 'running interface'		
		c = cfflib.load(self.inputs.cff_file)
		c_track = c.get_by_name(self.inputs.cff_track_name)			
		c_labels = c.get_by_name(self.inputs.cff_labels_name)
		c_network = c.get_connectome_network()[self.inputs.cff_network_number]
		"""
		(want to have something like this (above) for c_network, using 'get_by_name' 
		 as with c_track and c_labels above):
		if self.inputs.cff_network_name == None:
			c_network = c.get_by_name(self.inputs.cff_network_name)	
		else:
			c_network = c.get_by_name(self.inputs.cff_network_name)
		(...but not sure what to put for the '==None' bit (which doesn't work)		
		"""		
		c_track.load()
		c_labels.load()
		c_network.load()
		print 'loading fibres...'
		fibres_orig, hdr_orig = nib.trackvis.read(self.inputs.trk_file_orig, False)
		print str(len(fibres_orig)) + ' fibres loaded'	
		if not len(fibres_orig) == len(c_labels.data):
			print 'ERROR: TRK FILE IS NOT SAME SIZE AS FIBRE LABELS ARRAY' # (this needs to be an actual error exception that stops the function) 
		if self.inputs.ROI_xl_file2 == None:
			two_node_lists = 0
			ROI_list_dict1 = read_ROI_list(self.inputs.ROI_xl_file1)
			ROI_list_dict2 = ROI_list_dict1	
		elif not self.inputs.ROI_xl_file2 == None:
			two_node_lists = 1
			ROI_list_dict1 = read_ROI_list(self.inputs.ROI_xl_file1)
			ROI_list_dict2 = read_ROI_list(self.inputs.ROI_xl_file2)
		track_indices = []				
		for k in range(0, len(ROI_list_dict1.keys())):
			for kk in range(0,len(ROI_list_dict2.keys())):
				ROI1_name = str(ROI_list_dict1.values()[k])
				ROI1_number = int(ROI_list_dict1.keys()[k])
				ROI2_name = str(ROI_list_dict2.values()[kk])
				ROI2_number = int(ROI_list_dict2.keys()[kk])
				node_indices = [ROI1_number,ROI2_number]
				node_indices_reversed = [ROI2_number, ROI1_number]
				a = np.nonzero(c_labels.data==ROI1_number)[0]
				b = np.nonzero(c_labels.data==ROI2_number)[0]
				if ROI1_number in c_network.data.edge[ROI2_number]:
					n_fibs = c_network.data.edge[ROI2_number][ROI1_number]['number_of_fibers']
				elif ROI2_number in c_network.data.edge[ROI1_number]:
					n_fibs = c_network.data.edge[ROI1_number][ROI2_number]['number_of_fibers']
				else: n_fibs = 0
				if n_fibs>=self.inputs.n_fib_thresh:
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
			nib.trackvis.write(self.inputs.trk_file_new, outstreams, hdr_new)
		else:
			print 'no tracks found'		
		return runtime
	
	def _list_outputs(self):	
		outputs = self._outputs().get()
		fname = self.inputs.trk_file_new
		outputs["trk_file_new"] = fname
		return outputs


def read_ROI_list(ROI_xl_file):
	"""
	Reads in an excel file with two columns: 
	ROI numbers (left column) and ROI names (right column),
	and returns a dictionary that maps between the two
	
	Note - in the excel file the ROI list needs to start 
	from the 3rd row (first two rows are column headers)
	
	Usage: 
	
	ROI_list_dict = jg_nipype_interfacs.read_ROI_list(ROI_xl_file)
	
	"""
	import xlrd
	ROI_list_dict = {}
	wb = xlrd.open_workbook(ROI_xl_file)
	sh = wb.sheets()[0]
	for r in range(2, len(sh.col_values(0))):
		ROI_list_dict[int(sh.col_values(0)[r])] = sh.col_values(1)[r] 
	return ROI_list_dict		



class apply_QuickBundles_to_connectome_cnxns_InputSpec(BaseInterfaceInputSpec):

	QB_pkl_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store QB results', mandatory=False)	
	
	Ts_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Ts', mandatory=False)	
	Ets_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Ets', mandatory=False)	
	Vs_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Vs', mandatory=False)	

	cff_file = File(exists=True, desc='.cff (connectome file format file)', mandatory=True)
	cff_track_number =traits.Int(0, desc='number of track file in cff object. Default = 0', usedefault=True)
	#cff_track_name =traits.String('Tract file 0', desc='name of track file in cff object. Default = ''Tract file 0', usedefault=True)
	#cff_labels_name =traits.String('data_DT_pdfs_tracked_filtered_fiberslabel', desc='name of labels array in cff object. Default = ' 'data_DT_pdfs_tracked_filtered_fiberslabel',usedefault=True)	
	cff_labels_data_number =traits.Int(3, desc='number of labels data in cff object.')
	cff_network_number = traits.Int(0,desc='number (index - e.g. 0, 1, 2) of network file in cff object. Default is 0', usedefault=True)	
	QB_length_param =traits.Int(0, desc='number of track file in cff object. Default = 0', usedefault=True)
	QB_downsample_param = traits.Int(0, desc='number of track file in cff object. Default = 0,', usedefault=True)
	n_fib_thresh = traits.Int(1,desc='minimum number of fibres between node pairs', usedefault=True)
	
	
class apply_QuickBundles_to_connectome_cnxns_OutputSpec(TraitedSpec):
	
	QB_pkl_file_new =File(exists=True, desc="pkl_file_new", genfile=True) # Is the 'Mandatory' argument really necessary here?

	Ts_file_new =File(exists=True, desc="Ts_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?
	Ets_file_new=File(exists=True, desc="Ets_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?
	Vs_file_new=File(exists=True, desc="Vs_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?

		

class apply_QuickBundles_to_connectome_cnxns(BaseInterface):

	"""
	Usage:
	
	aQB = jg_nipype_interfaces.apply_QuickBundles_to_connectome_cnxns()
	aQB.inputs.cff_file = <'*.cff'>
	aQB.inputs.cff_track_name = 
	aQB.inputs.cff_labels_name = 
	aQB.inputs.cff_labels_data_number = 	
	aQB.inputs.QB_pkl_file_new = <'*.pkl'> 
	aQB.inputs.QB_downsample_param = 12
	aQB.inputs.QB_length_param = 4
	aQB.inputs.n_fib_thresh = 10
	aQB.run()
	
	Documentation here
	"""
	
	input_spec = apply_QuickBundles_to_connectome_cnxns_InputSpec
	output_spec = apply_QuickBundles_to_connectome_cnxns_OutputSpec
	
	def _run_interface(self,runtime):
		print 'running interface'	
		import time
		from nibabel import trackvis as tv		
		from dipy.tracking import metrics as tm
		from dipy.tracking import distances as td
		from dipy.io import pickles as pkl
		from dipy.viz import fvtk
		from dipy.segment.quickbundles import QuickBundles
			
		c = cfflib.load(self.inputs.cff_file)
		c_track = c.get_connectome_track()[self.inputs.cff_track_number]			
		c_labels = c.get_connectome_data()[self.inputs.cff_labels_data_number]
		c_network = c.get_connectome_network()[self.inputs.cff_network_number]
		c_track.load()
		all_tracks = c_track.data[0]
		hdr_orig = c_track.data[1]
		c_labels.load()
		
		Cs = []
		Ts = []
		skeletons = []
		orig_tracks = []
		qbs = []
		Vs = []
		Ets = []
		Eis = []
		Tpolys = []
		cnxn_list = []
			
		labels_mul = c_labels.data[:,0]*100000+c_labels.data[:,1]*10000000
		null_label = -1*100000+0*10000000 # this is the flag in the labels array for non-ROI-pair-connecting-fibres
		print 'null label = ' + str(null_label)
		#labels_unique_mul = labels.data[:,0]*100000+labels.data[:,1]*10000000
		val, inds = np.unique(labels_mul, return_inverse=True)
		# iterate through the list of unique values in the array
		for v in range(0,len(val)):
			inds_with_val_v = (np.nonzero(v==inds)[0]) # ...find all the entries with value v
			n_fibs = len(inds_with_val_v) # ...count how many
			if n_fibs > self.inputs.n_fib_thresh and not val[v]==null_label: # ...if the number of non-null-label elements with that value is large enough 
				print ' v =' + str(v)
				# ...Keep and do quickbundles on them: 
							# first make a record of the actual label (values of the 2 nodes) for this set of fibres
				# (all fibres corresponding to the entries in inds_with_val_v have the same value, so 
				# we just take the first one)
				ROI1 = c_labels.data[inds_with_val_v[0],0]
				ROI2 = c_labels.data[inds_with_val_v[0],1]
				
				cnxn_list.append([ROI1,ROI2])
				
				print 'cnxn with >' + str(self.inputs.n_fib_thresh) +' fibres: ROI ' + str(ROI1) + \
				      ' to ' + str(ROI2) + '(' + str(n_fibs) + ' fibres)'
				
				# now collect together all tracks for this cnxn			
				T = []
				for i in inds_with_val_v:
					T.append(all_tracks[i][0])
				
				#Downsample tracks to just 5 points:
				tracks=[tm.downsample(t,5) for t in T]
				
				#Perform Local Skeleton Clustering (LSC) with a 5mm threshold:
				now=time.clock()
				C=td.local_skeleton_clustering(tracks,d_thr=5)
				print('Done in %.2f s'  % (time.clock()-now,))
				
				Tpoly=[td.approx_polygon_track(t) for t in T]
				lens=[len(C[c]['indices']) for c in C]
				print('max %d min %d' %(max(lens), min(lens)))
				print('singletons %d ' % lens.count(1))
				print('doubletons %d' % lens.count(2))
				print('tripletons %d' % lens.count(3))
				
				skeleton=[]
			
				for c in C:			
					bundle=[Tpoly[i] for i in C[c]['indices']]
					si,s=td.most_similar_track_mam(bundle,'avg')    
					skeleton.append(bundle[si])
				
				for (i,c) in enumerate(C):    
					C[c]['most']=skeleton[i]
				for c in C:    
					print('Keys in bundle %d' % c)
					print(C[c].keys())
					print('Shape of skeletal track (%d, %d) ' % C[c]['most'].shape)
					
				qb = QuickBundles(Tpoly,self.inputs.QB_length_param,self.inputs.QB_downsample_param)
				V = qb.virtuals()
				Et, Ei = qb.exemplars() # Et = E tracks, Ei = E indices
				
				Cs.append(C)
				Ts.append(T)
				skeletons.append(skeleton)
				qbs.append(qb)
				Vs.append(V) # note: might want to put Vs and Es inside additional [] brackets...
				Ets.append(Et) # ...so that there 1 element ( len(V[X])=1 ) for each 
				Eis.append(Ei) # ...corresponding element in orig_tracks
				Tpolys.append(Tpoly)
				

		QB_names = ['QB_names_dict', 'Cs','Ts','skeletons','qbs','Vs','Ets','Eis', 'Tpolys', 'cnxn_list', 'hdr_orig']
		# want to also add the elements to the QB_data list in this loop, 
		# but not sure how to do that just yet	
		QB_names_dict = {}
		for q in range(0, len(QB_names)):
			QB_names_dict[QB_names[q]] = q			
		QB_data =  [ QB_names_dict,  Cs,  Ts,  skeletons,  qbs,  Vs,  Ets,  Eis,   Tpolys,   cnxn_list,   hdr_orig]
				
		if isdefined(self.inputs.QB_pkl_file_new):
			self._QB_pkl_file_new_path = self.inputs.QB_pkl_file_new
		else:
			self._QB_pkl_file_new_path = os.path.abspath("QB_pkl_file_new.pkl")
		pkl.save_pickle(self._QB_pkl_file_new_path,QB_data) # ...other outputs
		
		if isdefined(self.inputs.Ts_file_new):
			self._Ts_file_new_path = self.inputs.Ts_file_new
		else:
			self._Ts_file_new_path = os.path.abspath("Ts.pkl")
		pkl.save_pickle(self._Ts_file_new_path,Ts) # ...other outputs
		
		if isdefined(self.inputs.Ets_file_new):
			self._Ets_file_new_path = self.inputs.Ets_file_new
		else:
			self._Ets_file_new_path = os.path.abspath("Ets.pkl")
		pkl.save_pickle(self._Ets_file_new_path,Ets) # ...other outputs
		
		if isdefined(self.inputs.Vs_file_new):
			self._Vs_file_new_path = self.inputs.Vs_file_new
		else:		
			self._Vs_file_new_path = os.path.abspath("Vs.pkl")
		pkl.save_pickle(self._Vs_file_new_path,Vs) # ...other outputs
		
		return runtime
	
	def _gen_outfilename(self):
		_, name , _ = split_filename(self.inputs.cff_file)
		return name + "_QB" #Need to change to self.inputs.outputdatatype
	
	def _list_outputs(self):	
		outputs = self._outputs().get()
		
		outputs["QB_pkl_file_new"] = self._QB_pkl_file_new_path
		outputs["Ts_file_new"] = self._Ts_file_new_path
		outputs["Ets_file_new"] = self._Ets_file_new_path
		outputs["Vs_file_new"] = self._Vs_file_new_path
		return outputs


		"""
		for s in range(0, len(self.saved_file_names)):
			outputs[self.saved_file_names[s]] = self.saved_files[s]	
		"""
		
		"""
		if isdefined(self.inputs.QB_pkl_file_new):
			outputs["QB_pkl_file_new"] = self.inputs.QB_pkl_file_new 
		else:
			outputs["QB_pkl_file_new"] = os.path.abspath(self._gen_outfilename()) +'.pkl'
		

		if isdefined(self.inputs.Ts_file_new):
			outputs["Ts_file_new"] = self.inputs.Ts_file_new
		else:
			outputs["Ts_file_new"] = os.path.abspath(self._gen_outfilename()) +'_Ts.pkl'
		


		if isdefined(self.inputs.Ets_file_new):
			outputs["Ets_file_new"] = self.inputs.Ets_file_new
		else:
			outputs["Ets_file_new"] = os.path.abspath(self._gen_outfilename()) +'_Ets.pkl'


		if isdefined(self.inputs.Vs_file_new):
			outputs["Vs_file_new"] = self.inputs.Vs_file_new
		else:
			outputs["Vs_file_new"] = os.path.abspath(self._gen_outfilename()) +'_Vs.pkl'
		"""
		
		# need to add QB_all_data and QB_all_names lists to here

class apply_QuickBundles_to_trk_files_InputSpec(BaseInterfaceInputSpec):

	trk_files = traits.List(desc='list of .trk files to apply QB to')
	QB_pkl_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store QB results', mandatory=False)	
	QB_length_param =traits.Int(0, desc='quickbundles length parameter', usedefault=True)
	QB_downsample_param = traits.Int(0, desc='quickbundles downsample parameter', usedefault=True)
	
	
class apply_QuickBundles_to_trk_files_OutputSpec(TraitedSpec):
	
	QB_pkl_file_new =File(exists=False, desc="pkl_file_new", Mandatory=~False)

class apply_QuickBundles_to_trk_files(BaseInterface):

	"""
	Usage:
	
	aQB = jg_nipype_interfaces.apply_QuickBundles_to_connectome_cnxns()
	aQB.inputs.trk_files = < list of .trk filenames >
	aQB.inputs.QB_pkl_file_new = <'*.pkl'> 
	aQB.inputs.QB_downsample_param = 12
	aQB.inputs.QB_length_param = 4
	aQB.run()
	
	Documentation here
	"""
	
	input_spec = apply_QuickBundles_to_trk_files_InputSpec
	output_spec = apply_QuickBundles_to_trk_files_OutputSpec
	
	def _run_interface(self,runtime):
		print 'running interface'	
		import time
		from nibabel import trackvis as tv		
		from dipy.tracking import metrics as tm
		from dipy.tracking import distances as td
		from dipy.io import pickles as pkl
		from dipy.viz import fvtk
		from dipy.segment.quickbundles import QuickBundles
				
		# now collect together all tracks for this cnxn			
		T = []
		for t in self.inputs.trk_files:
			fibs, hdr = nib.trackvis.read(t)
			for f in fibs:
				T.append(f[0])
			
		#Downsample tracks to just 5 points:
		tracks=[tm.downsample(t,5) for t in T]
			
		#Perform Local Skeleton Clustering (LSC) with a 5mm threshold:
		now=time.clock()
		C=td.local_skeleton_clustering(tracks,d_thr=5)
		print('Done in %.2f s'  % (time.clock()-now,))
			
		Tpoly=[td.approx_polygon_track(t) for t in T]
		lens=[len(C[c]['indices']) for c in C]
		print('max %d min %d' %(max(lens), min(lens)))
		print('singletons %d ' % lens.count(1))
		print('doubletons %d' % lens.count(2))
		print('tripletons %d' % lens.count(3))
				
		skeleton=[]
			
		for c in C:			
			bundle=[Tpoly[i] for i in C[c]['indices']]
			si,s=td.most_similar_track_mam(bundle,'avg')    
			skeleton.append(bundle[si])
			
		for (i,c) in enumerate(C):    
			C[c]['most']=skeleton[i]
			
		for c in C:    
			print('Keys in bundle %d' % c)
			print(C[c].keys())
			print('Shape of skeletal track (%d, %d) ' % C[c]['most'].shape)
					
		qb = QuickBundles(Tpoly,self.inputs.QB_length_param,self.inputs.QB_downsample_param)
		V = qb.virtuals()
		Et, Ei = qb.exemplars() # Et = E tracks, Ei = E indices
				
		QB_names = ['QB_names_dict', 'C','T','skeleton','qb','V','Et','Ei', 'Tpoly', 'trk_files']
		# want to also add the elements to the QB_data list in this loop, 
		# but not sure how to do that just yet	
		QB_names_dict = {}
		for q in range(0, len(QB_names)):
			QB_names_dict[QB_names[q]] = q			

		QB_data =  [ QB_names_dict,  C,  T,  skeleton,  qb,  V,  Et,  Ei,   Tpoly, self.inputs.trk_files]
		
		pkl.save_pickle(self.inputs.QB_pkl_file_new,QB_data) # ...other outputs
		#print 'Saving Dictionary File to {path} in Pickle format'.format(path=op.abspath(self.dict_file))
		return runtime


	def _list_outputs(self):	
		outputs = self._outputs().get()
		fname = self.inputs.QB_pkl_file_new
		#outputs["trk_file_new"] = fname
		outputs["QB_pkl_file_new"] = fname
		
		# need to add QB_all_data and QB_all_names lists to here
		return outputs
	




class apply_flirt_to_fibs_InputSpec(BaseInterfaceInputSpec):
	
	in_fibs_file = traits.String(exists = True, desc="in fibs (pickle) file")
	input_type = traits.Enum('QB_struct', 'just_fibs', argstr='%s', desc='type of input - QB struct or just fibs')	
	flirt_transform_file=File(exists=True, desc="flirt transform file")
	DWIspace_image_file=File(exists=True, desc="source image used for flirt transform")
	refspace_image_file=File(exists=True, desc="target image used for flirt transform")
	flirted_fibs_file = traits.String(desc="new flirted fibs (pickle) file", Mandatory=False)
	 
class apply_flirt_to_fibs_OutputSpec(TraitedSpec):
	
	flirted_fibs_file =File(exists=True, desc="out fibs file")
	
class apply_flirt_to_fibs(BaseInterface):	
	"""
	Usage:
	"""

	input_spec = apply_flirt_to_fibs_InputSpec
	output_spec = apply_flirt_to_fibs_OutputSpec

	def _run_interface(self,runtime):
		
		from dipy.io import pickles as pkl
		from dipy.tracking.metrics import length
		from dipy.external.fsl import flirt2aff
		import numpy as np
		import nibabel as nib

		# load in flirt transform file
		dwi_img = nib.load(self.inputs.DWIspace_image_file)
		ref_img = nib.load(self.inputs.refspace_image_file)
		flirt_params = np.loadtxt(self.inputs.flirt_transform_file)
		aff = flirt2aff(mat=flirt_params, in_img=dwi_img,
	                  ref_img=ref_img)
		
		aff_inv = np.linalg.inv(aff)
	
		# load in in fibs file
		fibs_dat = pkl.load_pickle(self.inputs.in_fibs_file)
		
		if self.inputs.input_type == 'QB_struct':
			flirted_fibs = fibs_dat
			QB_names_dict = flirted_fibs[0]
			Ts = flirted_fibs[QB_names_dict['Ts']]
			Ets = flirted_fibs[QB_names_dict['Ets']]
			Vs = flirted_fibs[QB_names_dict['Vs']]
			ROI_labels = flirted_fibs[QB_names_dict['cnxn_list']]
			hdr_orig = flirted_fibs[QB_names_dict['hdr_orig']]		
			
			flirted_Ts = []
			for t in Ts:
				ft = [np.dot(tt, aff[:3,:3].T)+aff[:3,3] for tt in t]
				flirted_Ts.append(ft)			
			flirted_fibs[QB_names_dict['Ts']] = flirted_Ts
				
			flirted_Ets = []
			for e in Ets:
				et = [np.dot(ee, aff[:3,:3].T)+aff[:3,3] for ee in e]
				flirted_Ets.append(et)
			flirted_fibs[QB_names_dict['Ets']] = flirted_Ets
	
			flirted_Vs = []
			for v in Vs:
				vt = [np.dot(vv, aff[:3,:3].T)+aff[:3,3] for vv in v]
				flirted_Vs.append(vt)
			flirted_fibs[QB_names_dict['Vs']] = flirted_Vs
			
			
			"""
			Create new header 
			(check this gives same as applying affine to the DWIspace header?)
			
			new_hdr = nib.trackvis.empty_header()
			ref_aff = ref_img.get_affine()
			nib.trackvis.aff_to_hdr(ref_aff,new_hdr)
			new_hdr['dim'] = ref_img.shape
			flirted_fibs[QB_names_dict['hdr_orig']] = new_hdr # Need to change from 'hdr_orig' to just 'hdr'
			"""
			
			# Apply flirt transform to the trackvis hdr
			hdr_copy = hdr_orig.copy()
			aff_orig = nib.trackvis.aff_from_hdr(hdr_orig, atleast_v2=None)
			aff_new = np.dot(aff_orig,aff[:3,:3].T)+aff[:3,3]
			nib.trackvis.aff_to_hdr(aff_new,hdr_copy, pos_vox=None, set_order=None)
			hdr_copy['dim'] = ref_img.shape
			flirted_fibs[QB_names_dict['hdr_orig']] = hdr_copy # Need to change from 'hdr_orig' to just 'hdr'
			
			# NOTE: FLIRT OTHER THINGS - C, Ei, Tploy, Skeleton?
			#...need to make sure they are in the same fibre format
			
			# NOTE: change 'flirted fibs' to something closer / more generic?
			# In 'apply_QB_to_connectome_cnxns' it was called 'QB_data'
			
		elif self.inputs.input_type == 'just_fibs':			
			flirted_fibs = []
			flirted_fibs_tuple = [] # not used?
			for f in fibs_dat:
				flirted_f = [np.dot(ff, aff[:3,:3].T)+aff[:3,3] for ff in f]
				flirted_fibs.append(flirted_f)
				
		# write out flirted fibs to new pickle file
		if isdefined(self.inputs.flirted_fibs_file):
			self._flirted_fibs_file_path = self.flirted_fibs_file 
		else:
			self._flirted_fibs_file_path = os.path.abspath("flirted_fibs_file.pkl")
		pkl.save_pickle(self._flirted_fibs_file_path, flirted_fibs) # ...other outputs
				
		#print 'Saving Dictionary File to {path} in Pickle format'.format(path=op.abspath(self.dict_file))
		return runtime

	def _gen_outfilename(self): # NOT CURRENTLY USED
		_, name , _ = split_filename(self.inputs.in_fibs_file)
		return name + "_flirted_fibs" #Need to change to self.inputs.outputdatatype

	def _list_outputs(self):	
		outputs = self._outputs().get()
		#outputs["flirted_fibs_file"] = self._flirted_fibs_file_path
		outputs["flirted_fibs_file"] = self._flirted_fibs_file_path
		
		return outputs


class write_QuickBundles_to_trk_InputSpec(BaseInterfaceInputSpec):
	
	QB_pkl_file = traits.String(argstr = '%s', desc='name  of .pkl file containing QB results')	
	QB_output_type = traits.Enum('Ts_Ets_Vs', 'Ts', 'Ets', 'Vs', argstr='%s', desc='quickbundle output to write: ''Ts'', ''Ets'', ''Vs'', or ''Ts_Ets_Vs''') 
	#QB_all_data = traits.List()
	#QB_all_nanes = traits.List()
	#hdr_orig = traits.Array(desc='track file header, as a numpy array')
	outfile_name_stem = traits.String(argstr = '%s', desc='prefix of .trk file output')

class write_QuickBundles_to_trk_OutputSpec(TraitedSpec):
	
	outfile_list = traits.List(File, exists=True,  desc='list of .trk files output from this function')
	
class write_QuickBundles_to_trk(BaseInterface):
	"""
	write doc
	
	Ts is a 1xN bundles array  (i.e. len(O) = N)
	Vs and Ets are both 1x1 (i.e. len(V/D) = 1)
	
	indexing of Ts, Vs, and Es looks like this:
	O[streamlines][streamline_elements][X Y Z]
	
	'orig_hdr' is the hdr array from the connectome file
	( obtained using 'fib, hdr = nibabel.trackvis.read(file)' )
	
	"""		
	#	def write_QB_T_V_and_E_to_trk(T,V,E,hdr_orig,outfile_name):

	input_spec = write_QuickBundles_to_trk_InputSpec
	output_spec = write_QuickBundles_to_trk_OutputSpec
		
	def _run_interface(self,runtime):
		print 'running interface'		
		
		from dipy.io import pickles as pkl
		from nibabel import trackvis as tv
		
		QB_all = pkl.load_pickle(self.inputs.QB_pkl_file)
		
		Ts = QB_all[QB_all[0]['Ts']]
		Ets = QB_all[QB_all[0]['Ets']]
		Vs = QB_all[QB_all[0]['Vs']]
		ROI_labels = QB_all[QB_all[0]['cnxn_list']]
		hdr_orig = QB_all[QB_all[0]['hdr_orig']]		
		hdr_new = hdr_orig.copy()    	
		written_trk_files = []
		
		for r in range(0, len(Ts)):
			print 'track number ' + str(r) + ' of ' + str(len(Ts))
				
			trk_file_new = self.inputs.outfile_name_stem + str(ROI_labels[r][0]) + '_' + str(ROI_labels[r][1]) + '.trk'
			fib_new = []
			Ts_tuples = []
			Ets_tuples = []
			Vs_tuples = []
	
			if self.inputs.QB_output_type == 'Ts_Ets_Vs':
				hdr_new['n_scalars'] = np.array(1, dtype='int16')
				hdr_new['scalar_name'] = np.array(['track / virtual / exemplar','','', '', '', '', '', '', '', ''],dtype='|S20')
				hdr_new['n_properties'] = np.array(1, dtype='int16')
				hdr_new['property_name'] = np.array(['track / virtual / exemplar','','', '', '', '', '', '', '', ''],dtype='|S20')			
				
				# Numerical values serving as labels for the 3 different data types
				Ts_val = 1
				Ets_val = 2
				Vs_val = 3
				
				Ts_property_array = np.array([Ts_val], dtype='float32')
				Ets_property_array = np.array([Ets_val], dtype='float32')
				Vs_property_array = np.array([Vs_val], dtype='float32')
				
				# 
				for t in Ts[r]:	
					Ts_scalar_array = np.ones((len(t),1),dtype='float')*Ts_val
					new_tuple=tuple([t, Ts_scalar_array,Ts_property_array])
					#fib_new.append(new_tuple)
					Ts_tuples.append(new_tuple)

				Ets_scalar_array = np.ones((len(Ets[r][0]),1),dtype='float')*Ets_val
				new_tuple=tuple([Ets[r][0], Ets_scalar_array,Ets_property_array])
				Ets_tuples.append(new_tuple) 

				Vs_scalar_array = np.ones((len(Vs[r][0]),1),dtype='float')*Vs_val
				new_tuple=tuple([Vs[r][0], Vs_scalar_array,Vs_property_array])
				Vs_tuples.append(new_tuple)

				fib_new = Ts_tuples + Ets_tuples + Vs_tuples


				
			else:
				# Not using the property or scalar arrays for the simple Ts, Ets, and Vs
				hdr_new['n_scalars'] = np.array(0, dtype='int16')
				hdr_new['scalar_name'] = np.array(['','','', '', '', '', '', '', '', ''],dtype='|S20')
				hdr_new['n_properties'] = np.array(0, dtype='int16')
				hdr_new['property_name'] = np.array(['','','', '', '', '', '', '', '', ''],dtype='|S20')
				
				for t in Ts[r]:	
					new_tuple=tuple([t, None, None])
					#fib_new.append(new_tuple)
					Ts_tuples.append(new_tuple)
				
				new_tuple=tuple([Ets[r][0], None, None])
				Ets_tuples.append(new_tuple) 
				
				new_tuple=tuple([Vs[r][0], None, None])
				Vs_tuples.append(new_tuple)
				
				
				if self.inputs.QB_output_type == 'Ts': fib_new = Ts_tuples
				elif self.inputs.QB_output_type == 'Ets': fib_new = Ets_tuples
				elif self.inputs.QB_output_type == 'Vs': fib_new = Vs_tuples
			
			n_fib_out = len(fib_new)
			hdr_new['n_count'] = n_fib_out		
			tv.write(trk_file_new, fib_new, hdr_new)
			written_trk_files.append(os.path.abspath(trk_file_new))
		self._written_trk_files = written_trk_files
		
		return runtime
	
	def _list_outputs(self):	
		outputs = self._outputs().get() # outputs = self.output_spec().get()
		outputs['outfile_list'] = self._written_trk_files
		#outputs['outfile_list'] = self.outfile_list
		#outputs = self.output_spec().get()
		#for f in self.outfile_list:
		#	outputs['outfile_list'].append(f)
		return outputs







class apply_QuickBundles_to_QB_pkl_files_InputSpec(BaseInterfaceInputSpec):

	fib_type = traits.Enum('Ts', 'Ets', 'Vs', argstr='%s', desc='fibre type to use - Ts, Ets, or Vs')

	QB_pkl_file_list = traits.List(File,desc='list of QB pkl files')
	
	QB_pkl_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store QB results', mandatory=False)	

	
	Ts_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Ts', mandatory=False)	
	Ets_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Ets', mandatory=False)	
	Vs_file_new = traits.String(argstr = '%s', desc='name  of .pkl file to store Vs', mandatory=False)	

	QB_length_param =traits.Int(0, desc='number of track file in cff object. Default = 0', usedefault=True)
	QB_downsample_param = traits.Int(0, desc='number of track file in cff object. Default = 0,', usedefault=True)
	n_fib_thresh = traits.Int(1,desc='minimum number of fibres between node pairs', usedefault=True)
	
	
class apply_QuickBundles_to_QB_pkl_files_OutputSpec(TraitedSpec):
	
	QB_pkl_file_new =File(exists=True, desc="pkl_file_new", genfile=True) # Is the 'Mandatory' argument really necessary here?

	Ts_file_new =File(exists=True, desc="Ts_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?
	Ets_file_new=File(exists=True, desc="Ets_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?
	Vs_file_new=File(exists=True, desc="Vs_file_new",genfile=True) # Is the 'Mandatory' argument really necessary here?

		

class apply_QuickBundles_to_QB_pkl_files(BaseInterface):

	"""
	Documentation here
	"""
	
	input_spec = apply_QuickBundles_to_QB_pkl_files_InputSpec
	output_spec = apply_QuickBundles_to_QB_pkl_files_OutputSpec
	
	def _run_interface(self,runtime):
		print 'running interface'	
		import time
		from nibabel import trackvis as tv		
		from dipy.tracking import metrics as tm
		from dipy.tracking import distances as td
		from dipy.io import pickles as pkl
		from dipy.viz import fvtk
		from dipy.segment.quickbundles import QuickBundles
		
		
		all_tracks = [] # input parameter specifies whether these are Ts, Ets, or Vs
		all_inds = []
		
		# combine all subjects into one array
		for l in self.inputs.QB_pkl_file_list:
			QB_dat = pkl.load_pickle(l)
			QB_names_dict = QB_dat[0]
			all_tracks += QB_dat[QB_names_dict[self.inputs.fib_type]]
			all_inds += QB_dat[QB_names_dict['cnxn_list']]
		
		hdr_orig = QB_dat[QB_names_dict['hdr_orig']] # NEED TO RENAME THIS IN ALL FUNCTIONS TO JUST 'HDR'
		
		all_inds = np.array(all_inds)
		# find corresponding connections across subjecs
		labels_mul = all_inds[:,0]*100000+all_inds[:,1]*10000000
		val, inds = np.unique(labels_mul, return_inverse=True)
		# iterate through the list of unique values in the array
		Cs = []
		Ts = []
		skeletons = []
		orig_tracks = []
		qbs = []
		Vs = []
		Ets = []
		Eis = []
		Tpolys = []
		cnxn_list = []

		for v in range(0,len(val)):
			inds_with_val_v = (np.nonzero(v==inds)[0]) # ...find all the entries with value v
			n_fibs = len(inds_with_val_v) # ...count how many	
			if n_fibs >= self.inputs.n_fib_thresh:  # at the group level this will be n subjs for Ets and Vs and total numebr of fibs for Ts
				print ' v =' + str(v)
				ROI1 = all_inds[inds_with_val_v[0],0]
				ROI2 = all_inds[inds_with_val_v[0],1]				
				cnxn_list.append([ROI1,ROI2])
	
				# ...Keep and do quickbundles on them: 
						# first make a record of the actual label (values of the 2 nodes) for this set of fibres
				# (all fibres corresponding to the entries in inds_with_val_v have the same value, so 
				# we just take the first one)
				print 'cnxn with >' + str(self.inputs.n_fib_thresh) +' fibres: ROI ' + str(ROI1) + \
					      ' to ' + str(ROI2) + '(' + str(n_fibs) + ' fibres)'					
				
				# now collect together all tracks for this cnxn			
				T = []
				for i in inds_with_val_v:
					T.append(all_tracks[i][0])
				
				#Downsample tracks to just 5 points:
				tracks=[tm.downsample(t,5) for t in T]
				
				#Perform Local Skeleton Clustering (LSC) with a 5mm threshold:
				now=time.clock()
				C=td.local_skeleton_clustering(tracks,d_thr=5)
				print('Done in %.2f s'  % (time.clock()-now,))
				
				Tpoly=[td.approx_polygon_track(t) for t in T]
				lens=[len(C[c]['indices']) for c in C]
				print('max %d min %d' %(max(lens), min(lens)))
				print('singletons %d ' % lens.count(1))
				print('doubletons %d' % lens.count(2))
				print('tripletons %d' % lens.count(3))
				
				skeleton=[]
			
				for c in C:			
					bundle=[Tpoly[i] for i in C[c]['indices']]
					si,s=td.most_similar_track_mam(bundle,'avg')    
					skeleton.append(bundle[si])
				
				for (i,c) in enumerate(C):    
					C[c]['most']=skeleton[i]
				for c in C:    
					print('Keys in bundle %d' % c)
					print(C[c].keys())
					print('Shape of skeletal track (%d, %d) ' % C[c]['most'].shape)
					
				qb = QuickBundles(Tpoly,self.inputs.QB_length_param,self.inputs.QB_downsample_param)
				V = qb.virtuals()
				Et, Ei = qb.exemplars() # Et = E tracks, Ei = E indices
				
				Cs.append(C)
				Ts.append(T)
				skeletons.append(skeleton)
				qbs.append(qb)
				Vs.append(V) # note: might want to put Vs and Es inside additional [] brackets...
				Ets.append(Et) # ...so that there 1 element ( len(V[X])=1 ) for each 
				Eis.append(Ei) # ...corresponding element in orig_tracks
				Tpolys.append(Tpoly)
				

		QB_names = ['QB_names_dict', 'Cs','Ts','skeletons','qbs','Vs','Ets','Eis', 'Tpolys', 'cnxn_list', 'hdr_orig']
		# want to also add the elements to the QB_data list in this loop, 
		# but not sure how to do that just yet	
		QB_names_dict = {}
		for q in range(0, len(QB_names)):
			QB_names_dict[QB_names[q]] = q			
		QB_data =  [ QB_names_dict,  Cs,  Ts,  skeletons,  qbs,  Vs,  Ets,  Eis,   Tpolys,   cnxn_list, hdr_orig] # have removed 'hdr_orig'
				
		if isdefined(self.inputs.QB_pkl_file_new):
			self._QB_pkl_file_new_path = self.inputs.QB_pkl_file_new
		else:
			self._QB_pkl_file_new_path = os.path.abspath("QB_pkl_file_new.pkl")
		pkl.save_pickle(self._QB_pkl_file_new_path,QB_data) # ...other outputs
		
		if isdefined(self.inputs.Ts_file_new):
			self._Ts_file_new_path = self.inputs.Ts_file_new
		else:
			self._Ts_file_new_path = os.path.abspath("Ts.pkl")
		pkl.save_pickle(self._Ts_file_new_path,Ts) # ...other outputs
		
		if isdefined(self.inputs.Ets_file_new):
			self._Ets_file_new_path = self.inputs.Ets_file_new
		else:
			self._Ets_file_new_path = os.path.abspath("Ets.pkl")
		pkl.save_pickle(self._Ets_file_new_path,Ets) # ...other outputs
		
		if isdefined(self.inputs.Vs_file_new):
			self._Vs_file_new_path = self.inputs.Vs_file_new
		else:		
			self._Vs_file_new_path = os.path.abspath("Vs.pkl")
		pkl.save_pickle(self._Vs_file_new_path,Vs) # ...other outputs
		
		return runtime
	
	def _gen_outfilename(self):
		_, name , _ = split_filename(self.inputs.cff_file)
		return name + "_QB" #Need to change to self.inputs.outputdatatype
	
	def _list_outputs(self):	
		outputs = self._outputs().get()
		
		outputs["QB_pkl_file_new"] = self._QB_pkl_file_new_path
		outputs["Ts_file_new"] = self._Ts_file_new_path
		outputs["Ets_file_new"] = self._Ets_file_new_path
		outputs["Vs_file_new"] = self._Vs_file_new_path
		return outputs



