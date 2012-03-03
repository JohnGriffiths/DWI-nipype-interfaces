"""
==========================================================================
 jg_custom_interfaces.py - nipype wrappers for various jg python functions
==========================================================================

                                                         JDG 11/01/2012  
                                    
Contents:
---------

	- read ROI_geom
	- read_ROI_list	
	- make_trk_files_for_connectome_node_list
	- rewrite_trk_file_with_ED_vs_FL_scalars
	- apply_QuickBundles_to_connectome_cnxns
	- apply_QuickBundles_to_trk_files
	- apply_flirt_to_fibs
	- write_QuickBundles_to_trk
	- apply_QuickBundles_to_QB_pkl_files



	(NOTE: 'beta' version; not necessarily fully functioning atm)


"""

import os

import numpy as np

import cfflib

import nibabel as nib

from nipype.interfaces.cmtk.cmtk import length as fib_length

from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits,
                                    TraitedSpec, File, StdOutCommandLine,
                                    StdOutCommandLineInputSpec, BaseInterface,
                                     BaseInterfaceInputSpec, isdefined)

from nipype.utils.filemanip import split_filename

from dipy.io.pickles import save_pickle
import pickle as pkl 

from numpy import newaxis

from nipy.algorithms.utils.affines import apply_affine

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#import ipdb; ipdb.set_trace()
from IPython import embed

def flatten(l):
    """
    stolen from this website:
    http://lemire.me/blog/archives/2006/05/10/flattening-lists-in-python/
    see also:
    http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    """
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
    
    
def get_image_stats(filepath, thresh_list=None):
    """
    Add docue here
    
    THIS WORKS BUT IT ALWAYS GIVES 0 FOR THE PERCENT CALC. FIX
    """
    
    # Load in the image and do some calculations
    img = nib.load(filepath)
    img_dat = img.get_data()
    nz_voxels = img_dat[np.nonzero(img_dat)]
    img_stats_dict = {}
    img_stats_dict['num_voxels'] = img.shape[0]*img.shape[1]*img.shape[2]
    img_stats_dict['num_nz_voxels'] = len(nz_voxels)
    img_stats_dict['mean_intensity'] = np.average(img_dat)
    img_stats_dict['median_intensity'] = np.median(img_dat)
    img_stats_dict['mean_intensity_nz_voxels'] = np.average(nz_voxels)
    img_stats_dict['median_intensity_nz_voxels'] = np.median(nzvoxels)
    if thresh_list!=None:
        for t in thresh_list:
            voxels_gt_thresh = img_dat[np.nonzero(img_dat>t)]
            img_stats_dict['voxels_gt_' + str(t) + '_count'] = len(voxels_gt_thresh)
            img_stats_dict['voxels_gt_' + str(t) + '_pc_of_nz_voxels'] = (len(voxels_gt_thresh)/np.float(len(nz_voxels)))*100 # check this 
            img_stats_dict['voxels_gt_' + str(t) + '_mean_intensity'] = np.average(voxels_gt_thresh)
            img_stats_dict['voxels_gt_' + str(t) + '_median_intensity'] = np.median(voxels_gt_thresh)
    return img_stats_dict


def write_img_stats_to_csv(filenames,outfilename,labels=None,thresh_list=None):
    """
    SLIGHT PROBLEM = writes the rows out in a funny order...
    
    
    labels is an array of labels for each image
    thresh is a list of thresholds to be applied to the images
    
    
    I WANT TO BE ABLE TO DO THIS FOR VARIABLE NUMBERS OF VARLIST
    BUT CAN'T QUITE FIGURE OUT HOW
        - possibly use:
            - 'np.tile'
            - 'np.repeat'
            - 'itertools.combinations'
            - other itertools...        
    """
    """
    from xlwt import Workbook
    wb = Workbook()
    ws = wb.add_sheet('img_stats')
    """
    
    #order = filepath,labels,image_stats 
    output = []
    first_row = ['filename']
    if labels!=None:
        first_row.append([l for l in labels[0]]) # add the first entry in each label list    
        labels.pop(0) # remove that row now
    #[ws.write(1,l_iter,l[0]) for l_iter, l in enumerate(labels)]
    # now add the names of the values to get in img_stats
    img_stats_firstrow = get_image_stats(filenames[0],thresh_list)
    first_row.append([k for k in img_stats_firstrow.keys()])
    #[ws.write(1,k,img_stats_firstrow.keys()[k]) for k in range(0, len(img_stats_firstrow.keys()))]
    # add first_row to the output list
    output.append(flatten(first_row))
    # now get the data
    for f_it,f in enumerate(filenames):

        output_line = [f]       
        output_line.append([l for l in labels[f_it]])
        #[ws.write(f_it+1,l_it+l_iter,l[0]) for l_iter, l in enumerate(labels)]
        
        img_stats = get_image_stats(f,thresh_list)
        
        for i in img_stats.values():
            output_line.append(i)
            
        output.append(flatten(output_line))
        # add an empty line
        #output.append([])
       
    # write to a csv file
    #embed()
    f= open(outfilename, 'w')
    import csv
    outputfile = csv.writer(f)
    for line in output:
        outputfile.writerow(line)
    f.close()
    #wb.save(xl_filename)
     
            
  


def make_image_histograms(img_files,img_names=None,params_list=None, showfig=False,png_filename=None):
    """   
    Info
    ----
    Reads in a list of image filenames, and creates an nxm grid
    of histograms, where n is images and m is different types of
    histogram on the same image.
    
    Currently the default is to make three plots made for each image:
    
        1. Image histogram
        2. Image histogram for nonzero voxels only - # of voxels
        3. Image histogram for nonzero voxels only - % of voxels)
    
    To change from this default, provide a 'params_list' argument.
    
    (More functionality can be added if and when required.) 
    

    Inputs:
    -------
            Mandatory:
            
            image_file   - nifti/analyze image
            outfilename  - name of histogram png file
            
            
            Optional:
            
            params_list  - list of dictinoaries specifying how to plot the 
                           data in each column. Some options only require
                           the keyword argument, some also require a value
                           
                           Options:
                           
                           Key:                    Value:    Result:
                           
                           'only_nonzero_voxels'    ''        only plot nonzero voxels
                           'percent of voxels'      ''        label axes as % of voxels
            
            
            showfig      - binary; outputs the figure to screen before saving
            
            image_names  - list of names for the corresponding image files, used to label
                           each row in the histogram grid. Default is 'image 1', 'image2 2', etc.

 
    Outputs:
    --------
            histogram_png - '.png' file
    
    Example:
    -------
            
            make_image_histograms(<list of filenames>,showfig=True,png_filename=<filename>)
    """

    if params_list == None:
        params_list = [ dict([('', '')]),
                        dict([('only_nonzero_voxels','')]),
                        dict([('only_nonzero_voxels',''),('percent_of_voxels', '')])
                      ]
    fig = plt.figure()
    numrows = len(img_files)
    numcols = len(params_list)
    plotnum = 0
    for i_count,i in enumerate(img_files):
        if img_names==None:
            img_name='image ' + str(i_count)
        else:
            img_name = img_names[i_count]
        img_dat = nib.load(i).get_data()
        s = img_dat.shape
        d = img_dat.reshape(1,s[0]*s[1]*s[2]).flatten()
        # Make figure (taken from matplotlib 'hist' example	
        for p_count,p in enumerate(params_list):
            plotnum+=1
            #intstring = int(numrows+numcols+plotnum)
            ax = fig.add_subplot(str(numrows)+str(numcols)+str(plotnum))#intstring)

            # got this from (partially?) online book: 'Beginning Python visualization: crafting visual transformation scripts'
            bin_size = 0.2
            n_voxels = len(d)

            ax_title_end = ' voxels '
            
            if p.has_key('x_label'):
                xlabel=p['x_label']
            else:
                xlabel = 'image intensity'
            
            if p.has_key('only_nonzero_voxels'):
                d = d[np.nonzero(d)]
                ax_title_end = '(nonzero) voxels'
                
            if p.has_key('percent_of_voxels'):
                prob, bins, patches = ax.hist(d,bins=50, normed=True)#np.arange(0,max(d)+bin_size,bin_size), normed=True)#,align='center')
                """
                (alternative way of doing the histogram):
                prob, bins = np.histogram(d,normed=True,bins=200)#,align='center')
                width = 0.7*(bins[1]-bins[0])
                center=(bins[:-1]+bins[1:])/2
                ax.bar(center, prob*np.diff(bins)*100,align='center',width=width)
                """
                ax_title_start = '% of '
            else:
                prob, bins, patches = ax.hist(d,bins=50)#np.arange(0,max(d)+bin_size,bin_size))#,align='center')
                ax_title_start = '# of '
            
            ax_title=ax_title_start+ax_title_end               
            ax.tick_params(labelsize='small')
            ax.grid(True)
            #ax.set_ylabel(ylabel, fontsize='small')
            if i_count==0:#i_count==len(img_dat_list)-1:                
                ax.set_title(ax_title, fontsize='small')
            if i_count==len(img_files)-1:#0:
                ax.set_xlabel(xlabel, fontsize='small')
            if p_count==0:
                ax.set_ylabel(img_name)
    if showfig==True:
        plt.show()
    if png_filename!=None:
        print 'writing histogram to   ' + png_filename
        plt.savefig(png_filename, format="png")
    #return plt

    


class make_image_histogram_fig_InputSpec(BaseInterfaceInputSpec):
    
    image_file = traits.File(desc='image file') 
    # (I want to use this as a list but also need it to take single files so am leaving it as the latter for now )
	#image_files = traits.List(File,desc='image file list',argstr='%s')
    outfilename = traits.String(argstr = '%', desc='histogram png filename', mandatory=False)

class make_image_histogram_fig_OutputSpec(TraitedSpec):
    
	histogram_png = traits.File(desc="histogram png file",exists=False)

class make_image_histogram_fig(BaseInterface):
    """   
    Info
    ----
    Wraps function 'make_image_histograms' that calculates histograms 
    for a series of input images and outputs them as .png files
    
    The 'make_image_histograms' function has more flexibility on what 
    it plotted, but this nipype wrapper (currently) outputs a single 
    .png file with nx3 subplots, where n is number of input images. 

    The three plots made for each image are:
    
        1. Image histogram
        2. Image histogram for nonzero voxels only - # of voxels
        3. Image histogram for nonzero voxels only - % of voxels)
    
    More functionality can be added if and when required. 
    

    Inputs:
    -------
            image_file   - nifti/analyze image
            outfilename  - name of histogram png file
 
    Outputs:
    --------
            histogram_png - '.png' file
    
    Example:
    -------
            mihf = jg_custom_interfaces.make_image_histogram_fig()
	        mihf.inputs.image_file = <list of filenames>
	        mihf.inputs.outfilename = <filename>
	        mihf.run()
    """
    input_spec = make_image_histogram_fig_InputSpec
    output_spec = make_image_histogram_fig_OutputSpec

    def _run_interface(self,runtime):
        self._outfilename = self._gen_outfilename()
        print 'calculating histogram'
        make_image_histograms([self.inputs.image_file],png_filename=self._outfilename)
        """
        Alternative = get the 'plt' object from the function:
            plt = make_image_histograms(self.inputs.image_file,png_filename=self._outfilename)
            print 'writing png file'
            plt.savefig(self._outfilename)    
        """
        return runtime
    
    def _gen_outfilename(self):
        if isdefined(self.inputs.outfilename):
            fname = self.inputs.outfilename
        else:
            fname = os.path.abspath('image_histogram.png')
        return fname
    
    def _list_outputs(self):    
        outputs = self._outputs().get()
        outputs["histogram_png"] = self._outfilename
        return outputs
















def get_ROI_geom(img,dat=None): #indices,voxel_dimensions, label):
    """
    [ IF THIS TAKES AGES FOR LAST VOLUMES, TRY VECTORIZING CODE...]
    [ YES I THINK IT'S TAKING TOO LONG - NEED TO TRY THAT ]
    [add proper doc]
    'dat' is an optional argument that, when specified, 
    is used to provide the data array, rather than 
    extracting it from the img object. So the dat argument
    can be used when you want to produce an ROI from an
    from an image within python on the fly without writing
    it out to file
    """
    if dat == None:
        img_dat = img.get_data()
    else:
        img_dat = dat 
    inds = np.nonzero(img_dat)
    voxdims = img.get_header().get_zooms()
    nvox = len(inds[0]) # number of voxels
    print '\n     # voxels = ' + str(nvox) 
    vol = len(inds[0])*(voxdims[0]*voxdims[1]*voxdims[2])
    print '\n volume (mm3) = ' + str(vol)
    geom_dict = dict([['n_voxels', nvox], ['volume', vol]])
    diam_dict = get_ROI_diameter(img, img_dat) 
    geom_dict.update(diam_dict)
    print '\n     diameter = ' + str(geom_dict['diameter'])                                        
    print '       ...calculated between '
    print '          coord_1: ' + str(geom_dict['coord1'])
    print '          coord_2: ' + str(geom_dict['coord2'])
    return geom_dict

def get_ROI_diameter(img, dat=None,list_extremas=False, make_extrema_box=False):
    """
    [add proper doc]
    'dat' is an optional argument that, when specified, 
    is used to provide the data array, rather than 
    extracting it from the img object. So the dat argument
    can be used when you want to produce an ROI from an
    from an image within python on the fly without writing
    it out to file
    
    include 'list_extremas=True' to include the image extremas
    in the returned dictionary    
    """
    print ' WARNING: HAVENT YET CONFIRMED THAT GET_ROI_DIAMETER WORKS!!!'
    
    if dat==None:
        img_dat = img.get_data()
    else:
        img_dat = dat 
    img_dat_rs = np.reshape(img_dat,[img_dat.shape[0]*img_dat.shape[1]*img_dat.shape[2]])
    
    img_coords = get_img_coords(img)    

    img_coords_dim1_rs = np.reshape(img_coords[:,:,:,0],[1,img_coords.shape[0]*img_coords.shape[1]*img_coords.shape[2]])
    img_coords_dim2_rs = np.reshape(img_coords[:,:,:,1],[1,img_coords.shape[0]*img_coords.shape[1]*img_coords.shape[2]])
    img_coords_dim3_rs = np.reshape(img_coords[:,:,:,2],[1,img_coords.shape[0]*img_coords.shape[1]*img_coords.shape[2]])    
    img_coords_dim123_rs = np.squeeze(np.array([img_coords_dim1_rs, img_coords_dim2_rs, img_coords_dim3_rs]))
    
    img_coords_rs = np.reshape(img_coords, [len(img_dat_rs),3])
    #img_coords_rs = np.reshape(img_coords,[1,img_coords.shape[0]*img_coords.shape[1]*img_coords.shape[2]])
#    img_dat_rs = np.reshape(img_dat,[1,img_dat.shape[0]*img_dat.shape[1]*img_dat.shape[2]])

    img_dat_rs_nz_inds = np.nonzero(img_dat_rs)
    #img_coords_dim123_rs_nz = img_coords_dim123_rs[:,img_dat_rs_nz_inds]
    img_coords_rs_nz = img_coords_rs[img_dat_rs_nz_inds]

    """
    nz_inds = np.nonzero(img_dat_rs)                                     
    nz_img_coords = img_coords_rs[nz_inds] # coordinates of non-zero voxels; also is an n voxels x3 array
    """
    
    extremas = []
    extremas_nz = []
    all_maxvalinds = []
    all_minvalinds = []
    all_maxvalinds_nz = []
    all_minvalinds_nz = []
    
    extremas_list = [['max', 'min', 'diff']]
     # to get back the actual voxel values:
    # iterate through each image dimension
    for n in [0,1,2]:
        # find the outermost nonzero coordinate
        #maxval= max(img_coords_dim123_rs[n,:])
        maxval= max(img_coords_rs_nz[:,n])
        
        # find the indices of elements in the in nz_img_coords
        # that have that value (for the current dimension)
        #maxvalinds = np.nonzero(img_coords_dim123_rs[n,:]==maxval)
        maxvalinds = np.nonzero(img_coords_rs[:,n]==maxval)[0]
        maxvalinds_nz = np.nonzero(img_coords_rs_nz[:,n]==maxval)[0]

        # add to the extremas and nz_img_coord indices (maxvalinds) lists
        #[extremas.append(mv) for mv in np.squeeze(img_coords_dim123_rs[:,maxvalinds]).T]
        [extremas.append(mv) for mv in img_coords_rs[maxvalinds]]                                                     
        [extremas_nz.append(mv) for mv in img_coords_rs_nz[maxvalinds_nz]]                                                     
        
        all_maxvalinds+=list(maxvalinds)
        all_maxvalinds_nz+=list(maxvalinds_nz)
        
        # do the same for outermost nonzero voxel in the -ve direction (minima)
        #minval= min(img_coords_dim123_rs[n,:])
        minval= min(img_coords_rs_nz[:,n])

        #minvalinds = np.nonzero(img_coords_dim123_rs[n,:]==minval)
        minvalinds = np.nonzero(img_coords_rs[:,n]==minval)[0]
        minvalinds_nz = np.nonzero(img_coords_rs_nz[:,n]==minval)[0]

        #[extremas.append(mv) for mv in np.squeeze(img_coords_dim123_rs[:,minvalinds]).T]
        [extremas.append(mv) for mv in img_coords_rs[minvalinds]]                                                     
        [extremas_nz.append(mv) for mv in img_coords_rs_nz[minvalinds_nz]]                                                     

        all_minvalinds+=list(minvalinds)
        all_minvalinds_nz+=list(minvalinds_nz)
        
        extremas_list.append([maxval, minval, maxval+abs(minval)]) 
        # (the difference only works if the minima have '-' coordinates
        
        if n==0:
            n0_maxvalinds = maxvalinds
            n0_minvalinds = minvalinds
        if n==1:
            n2_maxvalinds = maxvalinds
            n2_minvalinds = minvalinds
        if n==2:
            n2_maxvalinds = maxvalinds
            n2_minvalinds = minvalinds

    
    diam = 0    
    for c1 in extremas_nz:
        for c2 in extremas_nz:
            eudist = sum(np.sqrt((c1-c2)**2))
            if eudist>diam:
                diam=eudist
                coord1=c1
                coord2=c2
    diam_dict = dict([['diameter', diam], ['coord1', coord1], ['coord2', coord2]])
    if list_extremas==True:
        diam_dict['extremas'] = extremas
        diam_dict['extremas_nz'] = extremas_nz
        diam_dict['extremas_list'] = extremas_list
    if make_extrema_box == True:
        new_img_dat = np.zeros(img_dat_rs.shape)
        new_img_dat[all_minvalinds] = 1 #np.ones(len(all_minvalinds))
        new_img_dat[all_maxvalinds] = 2 #np.ones(len(all_minvalinds))*2
        # reshape new_img_dat back to the original img array size
        new_img_dat_rs = np.reshape(new_img_dat,img_dat.shape)
        """
          check for this:
          sh =img_dat.shape 
          diff = np.reshape(new_img_dat,sh)-np.reshape(img_dat_rs,sh)
          np.nonzero(diff)  --> empty array 
        """
        new_img = nib.Nifti1Image(new_img_dat_rs,img.get_affine())
      
        return diam_dict, new_img
    else:
        return diam_dict

def get_img_coords(img):
    """
    [add proper doc]
    courtesy of E.Garyfallidis
    """
    print ' WARNING: HAVENT YET CONFIRMED THAT GET_IMG_COORDS WORKS!!!'
    i,j,k = img.shape
    coords = np.zeros([i,j,k,3])
    coords[...,0] = np.arange(i)[:,newaxis,newaxis]
    coords[...,1] = np.arange(j)[newaxis,:,newaxis]
    coords[...,2] = np.arange(k)[newaxis,newaxis,:]
    aff = img.get_affine()
    img_mm = apply_affine(aff, coords)
    return img_mm


class read_ROI_geom_InputSpec(BaseInterfaceInputSpec):
    ROI_file = traits.File(exists=True, desc='ROI image file',argstr='%s')
    use_labels = traits.Bool(argstr='%s',desc='use labels')
    output_type = traits.Enum('pickled_dict', 'txt_file', 'screen_only', argstr='%s',desc='type of output')
    outfilename = traits.String(argstr = '%', desc='output ROI geometry file', mandatory=False)

class read_ROI_geom_OutputSpec(TraitedSpec):
	ROI_geom_list = traits.File(desc="ROI geometry info file",exists=False)

class read_ROI_geom(BaseInterface):
    """   
    Info
    ----
   
    Calculates basic geometric properties (number of voxels, volume, [others?])
    of regions of interest (ROIs) in a nifti / analyze image. When a labelled 
    image (i.e. one with multiple values defining sub-regions), and the 'use_labels'
    input argument is set to True, geometric properties are calculated for each 
    unique voxel value in the image. Otherwise, it is assumed that the image is 
    a single ROI (potentially with multiple voxel values), and geometries are 
    calculated once for all nonzero voxels together.Outputs can be to screen, to
     .txt file, or a numpy array
    
    Inputs:
    -------
            ROI_file     - nifti/analyze image
            use_labels   - Boolean 
            output_type  - 'pickled_dict', 'txt_file', or 'screen_only'
            out_file     - name of output file (generated automatically from 
                           ROI_file if left blank )
            
    Outputs:
    --------
            ROI_geom_list - '.npy' or '.txt' file if selected
    
    Example:
    -------
            get_geom = jg_custom_interface.read_ROI_geom()
            get_geom.inputs.ROI_file = <filename>
            get_geom.inputs.use_labesls = True
            get_geom.inputs.output_type = 'pickled_dict'
            get_geom.inputs.outfilename = <filename>
            get_geom.run()   
    """
    input_spec = read_ROI_geom_InputSpec
    output_spec = read_ROI_geom_OutputSpec

    def _run_interface(self,runtime):
            
            img = nib.load(self.inputs.ROI_file)
            img_dat = img.get_data()
            blank_img = np.zeros(img_dat.shape)
            vox_dims = img.get_header().get_zooms()
            
            
            if self.inputs.use_labels==False:
                labels = 'all_nonzero'
            else:
                labels = list(np.unique(img_dat))
                labels.remove(0)  # remove label '0'
            
            geoms_dict_all_labels = {}
            #geoms_array = np.zeros([len(labels),6])
            geoms_list_all_labels = []
           
            if labels=='all_nonzero':
                inds = np.nonzero(img_dat)	
                new_img_dat = np.zeros(img_dat.shape)
                new_img_dat[inds] = np.ones(len(inds[0]))
                geoms_dict = get_ROI_geom(img, new_img_dat) 
	        geoms_dict_all_labels['all_nonzero'] = geoms_dict
            else:
                for l in range(0, len(labels)):  
                    print '\n label = ' + str(labels[l]) 
                    inds = np.nonzero(img_dat==labels[l])
                    new_img_dat = np.zeros(img_dat.shape) 
                    new_img_dat[inds] = np.ones(len(inds[0]))
                    geoms_dict = get_ROI_geom(img, new_img_dat)
                    geoms_dict_all_labels[labels[l]] = geoms_dict
                
                #geoms_array[l+1,0] = labels[l]
                #geoms_array[l+1,1:] = np.array(geoms_dict.values())
                geoms_list = geoms_dict.values()
                geoms_list.insert(0,l)
                geoms_list_all_labels.append(geoms_list)
                #for gd in geoms_dict.keys():
                #    geoms_dict_all_labels[l,gd] = geoms_dict[gd]

            geoms_list_headers = geoms_dict.keys()
            geoms_list_headers.insert(0, 'label')
            geoms_list_all_labels.insert(0, geoms_list_headers)
            #geoms_array[0,0] = 'label'
            #geoms_array[0,1:] = np.array(geoms_dict.keys())
            
            outfilename = self._gen_outfilename()                            
            
            geoms_all = dict([['dict',geoms_dict_all_labels],['list', geoms_list_all_labels]])
            
            print '\n saving to ' + outfilename + '\n'
            if self.inputs.output_type == 'pickled_dict':
                save_pickle(outfilename, geoms_all)
                f = open(outfilename, 'w')
                f.writelines(["%s\n" % str(gl) for gl in geoms_list])
                #f.writelines(list( '%s \n ' % gl for gl in geoms_list))
                f.close()           
            elif self.inputs.output_type == 'screen_only':
                print 'output to screen only - no file written'
               
            return runtime
    
        
                
    def _gen_outfilename(self):
        if isdefined(self.inputs.outfilename):
            fname = self.inputs.outfilename
        else:
            _, name , _ = split_filename(self.inputs.ROI_file)
            if self.inputs.output_type == 'pickled_dict':
                fname = name + '_pickled_dict.pkl'
            elif self.inputs.output_type == 'txt_file':
                fname = name+'_ROI_geom.txt'
        return fname
    
    def _list_outputs(self):    
        outputs = self._outputs().get()
        fname = self._gen_outfilename()
        outputs["ROI_geom_list"] = fname
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


class make_trk_files_for_connectome_node_list_InputSpec(BaseInterfaceInputSpec):
    
    ROI_xl_file1 = File(exists=True, desc='excel file with list of node ROI numbers to identify', mandatory=True)
    ROI_xl_file2 = File(exists=True, desc='second excel file with list of node ROI numbers to identify', mandatory=False)
    cff_file = File(exists=True, desc='.cff (connectome file format file)', mandatory=True)

class make_trk_files_for_connectome_node_list_InputSpec(BaseInterfaceInputSpec):
    
    ROI_xl_file1 = File(exists=True, desc='excel file with list of node ROI numbers to identify', mandatory=True)
    ROI_xl_file2 = File(exists=True, desc='second excel file with list of node ROI numbers to identify', mandatory=False)
    cff_file = File(exists=True, desc='.cff (connectome file format file)', mandatory=True)
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


def rewrite_trk_file_with_ED_vs_FL_scalars(trk_file_orig,scalar_type,trk_file_new=None, lengths_list_file_new=None):
    """
    (see nipype version for documentation and example)
    
    ( NOTE: haven't tested since modifying to separate nipype and raw function )
    
    trk_file_orig can be either a filename (string) or a [fib,hdr] pair (list), 
    where the elements are those given by nibabel when reading in trackvis files
    ...the point of the latter is so that .trk files can be read in separately and 
    modified if necessary (e.g. masks applied) on-the-go in python without having
    to write out to more files  

    """
    if type(trk_file_orig) == str and os.path.isfile(trk_file_orig):        
        print 'loading fibres...'
        fib_orig, hdr_orig = nib.trackvis.read(trk_file_orig, False)
        print str(len(fib_orig)) + ' fibres loaded'
    elif type(trk_file_orig) == list:
        print 'fibres passsed directly...'
        fib_orig =trk_file_orig[0]
        hdr_orig = trk_file_orig[1]
    hdr_new = hdr_orig.copy()
    fib_new = []
    lengths_list = [['Fibre Length (FL)', 'Euclidean Distance (ED)', 'FL minus ED', 'ED as a percentage of FL']]
    for f in fib_orig:
        # Calculate fibre lengths    
        FL = fib_length(f[0]) 
        # Calculate Euclidean distance between fibre start and endpoints
        ED = np.sqrt(np.square(f[0][0][0]-f[0][-1][0])+np.square(f[0][0][1]-f[0][-1][1])+np.square(f[0][0][2]-f[0][-1][2]))
        # Calculate fibre length minus Euclidean distance:
        FL_sub_ED = np.subtract(FL, ED)
        # Calculate Euclidean distance as a percentage of fibre length
        ED_pco_FL = np.divide(100,FL)*ED
        
        lengths_list.append([FL, ED, FL_sub_ED, ED_pco_FL])
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
    if trk_file_new!=None:
        nib.trackvis.write(trk_file_new, fib_new, hdr_new)        
    if lengths_list_file_new!=None:
        save_pickle(lengths_list_file_new, lengths_list)
    return fib_new, hdr_new, lengths_list


class rewrite_trk_file_with_ED_vs_FL_scalars_nipype_InputSpec(BaseInterfaceInputSpec):
    
    trk_file_orig = File(exists=True, desc='original track file', mandatory=True)
    trk_file_new = traits.String(argstr = '%s', desc='name of new track file to be made', mandatory=True)
    lengths_list_file_new = traits.String(argstr = '%s', desc='name of new lengths list .pkl file to be made', mandatory=True)
    scalar_type = traits.String(argstr='%s',desc='Type of scalar...', mandatory=True)		

class rewrite_trk_file_with_ED_vs_FL_scalars_nipype_OutputSpec(TraitedSpec):
    
    trk_file_new = File(exists=True, desc="trk_file_new")
    lengths_list_file_new = File(exists=False, desc="length_stats")

	
class rewrite_trk_file_with_ED_vs_FL_scalars_nipype(BaseInterface):
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
		r = jg_nipype_interfaces.rewrite_trk_file_with_ED_vs_FL_scalars_nipype()
		r.inputs.trk_file_orig = <trk_file_orig>
		r.inputs.trk_file_new  = <trk_file_new>
		r.inputs.lengths_list_file_new  = <lengths_list_file_new>
		r.inputs.scalar_type   = <scalar_type>
		r.run()
	 
	Inputs:
	
		trk_file_orig - original .trk file
		trk_file_new  - name of new .trk file to be written 
		scalar_type   - type of scalar to write ('FL', 'ED', 'FL_sub_ED', 'ED_pco_FL'; see above)
	"""
    input_spec = rewrite_trk_file_with_ED_vs_FL_scalars_nipype_InputSpec
    output_spec = rewrite_trk_file_with_ED_vs_FL_scalars_nipype_OutputSpec	
    def _run_interface(self,runtime):
        scalar_type = self.inputs.scalar_type
        trk_file_orig = self.inputs.trk_file_orig
        trk_file_new = self.inputs.trk_file_new
        lengths_list_file_new = self.inputs.lengths_list_file_new
        fib_new, hdr_new, lengths_list = rewrite_trk_file_with_ED_vs_FL_scalars(
                                         trk_file_orig, scalar_type, trk_file_new=trk_file_new,
                                         lengths_list_file_new=lengths_list_file_new)
        return runtime
    def _list_outputs(self):
        outputs = self._outputs().get()
        tfn = self.inputs.trk_file_new
        llfn = self.inputs.lengths_list_file_new
        outputs["trk_file_new"] = tfn
        outputs["lengths_list_file_new"] = llfn
        return outputs

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

	hdr_orig = hdr # when a list of trks is used, just takes the header of the last in the list	
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
	
        cnxn_list = [['had_to_add_a_name_here_to_get_the_interface_to_work1', 'ibid']]		       
        #QB_names = ['QB_names_dict', 'Cs','Ts','skeletons','qbs','Vs','Ets','Eis', 'Tpolys', 'cnxn_list', 'hdr_orig']       
        QB_names = ['QB_names_dict', 'C','T','skeleton','qb','V','Et','Ei', 'Tpoly', 'cnxn_list', 'hdr_orig', 'trk_files']
        # want to also add the elements to the QB_data list in this loop, 
        # but not sure how to do that just yet	
        QB_names_dict = {}
        for q in range(0, len(QB_names)):
            QB_names_dict[QB_names[q]] = q
        QB_data =  [ QB_names_dict,  C,  T,  skeleton,  qb,  V,  Et,  Ei,   Tpoly, cnxn_list, hdr_orig, self.inputs.trk_files]
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
	[ add doc ] 
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
			
			# Apply flirt transform to the trackvis hdr
            # NOTE: STILL NEED TO CHECK THE .TRK FILES FROM THIS
			hdr_copy = hdr_orig.copy()
			nib.trackvis.aff_to_hdr(aff,hdr_copy)#, pos_vox=None, set_order=None)
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
			self._flirted_fibs_file_path = self.inputs.flirted_fibs_file 
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
	[write doc]
	
	Ts is a 1xN bundles array  (i.e. len(O) = N)
	Vs and Ets are both 1x1 (i.e. len(V/D) = 1)
	
	indexing of Ts, Vs, and Es looks like this:
	O[streamlines][streamline_elements][X Y Z]
	
	'orig_hdr' is the hdr array from the connectome file
	( obtained using 'fib, hdr = nibabel.trackvis.read(file)' )
	
	"""		
	input_spec = write_QuickBundles_to_trk_InputSpec
	output_spec = write_QuickBundles_to_trk_OutputSpec
		
	def _run_interface(self,runtime):
		print 'running interface'		
		
		from dipy.io import pickles as pkl
		from nibabel import trackvis as tv
		
		QB_all = pkl.load_pickle(self.inputs.QB_pkl_file)
		
		Ts = QB_all[QB_all[0]['T']] # temporary: changed from Ts']]
		Ets = QB_all[QB_all[0]['Et']] # temporary: changed from Et']]
		Vs = QB_all[QB_all[0]['V']] # temporary: changed from V']]
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





def write_QuickBundles_to_trk_alternative(QB_pkl_file, trk_file_new, QB_output_type):
	"""
	Alternative to the original nipype interface 
	for this (above), because it wasn't working on 
	outputs from the apply_QuickBundles_to_trk' interface
	 - need to sort out the differences between these
	 at some point; at the moment though just need a working
	 function for this
	 
	 Isn't doing the Et/V/T combination properly;
	 but will output them separately...
	 
	 """
	from dipy.io import pickles as pkl
	from nibabel import trackvis as tv
	QB_all = pkl.load_pickle(QB_pkl_file)
	Ts = QB_all[QB_all[0]['T']] # temporary: changed from Ts']]
	Ets = QB_all[QB_all[0]['Et']] # temporary: changed from Et']]
	Vs = QB_all[QB_all[0]['V']] # temporary: changed from V']]
	#ROI_labels = QB_all[QB_all[0]['cnxn_list']]
	hdr_orig = QB_all[QB_all[0]['hdr_orig']]		
	hdr_new = hdr_orig.copy()
	fib_new = []
	Ts_tuples = []
	Ets_tuples = []
	Vs_tuples = []
	if QB_output_type == 'Ts_Ets_Vs':
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
		for t in Ts:	
			Ts_scalar_array = np.ones((len(t),1),dtype='float')*Ts_val
			new_tuple=tuple([t,Ts_scalar_array,Ts_property_array])
			Ts_tuples.append(new_tuple)
	
		Ets_scalar_array = np.ones((len(Ets[0]),1),dtype='float')*Ets_val
		new_tuple=tuple([Ets[0], Ets_scalar_array,Ets_property_array])
		Ets_tuples.append(new_tuple) 
		
		Vs_scalar_array = np.ones((len(Vs[0]),1),dtype='float')*Vs_val
		new_tuple=tuple([Vs[0], Vs_scalar_array,Vs_property_array])
		Vs_tuples.append(new_tuple)
		fib_new = Ts_tuples + Ets_tuples + Vs_tuples
	else: # Not using the property or scalar arrays for the simple Ts, Ets, and Vs
		hdr_new['n_scalars'] = np.array(0, dtype='int16')
		hdr_new['scalar_name'] = np.array(['','','', '', '', '', '', '', '', ''],dtype='|S20')
		hdr_new['n_properties'] = np.array(0, dtype='int16')
		hdr_new['property_name'] = np.array(['','','', '', '', '', '', '', '', ''],dtype='|S20')
		for t in Ts:	
			new_tuple=tuple([t, None, None])
			Ts_tuples.append(new_tuple)
		
		new_tuple=tuple([Ets[0], None, None])
		Ets_tuples.append(new_tuple)
		
		new_tuple=tuple([Vs[0], None, None])
		Vs_tuples.append(new_tuple)
		
		if QB_output_type == 'Ts': fib_new = Ts_tuples
		elif QB_output_type == 'Ets': fib_new = Ets_tuples
		elif QB_output_type == 'Vs': fib_new = Vs_tuples
	
	n_fib_out = len(fib_new)
	hdr_new['n_count'] = n_fib_out		
	tv.write(trk_file_new, fib_new, hdr_new)
	




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



"""
Function for producing image histograms

inputs:
	images
	args:
		- nonzero
		- range
		- nbins
		

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import nibabel as nib

img_file = '/work/imaging5/DTI/CBU_DTI/speckles_tester/john_nipype_camino_workflow/64D_2A/speckles_check_workflow/data_analyses/_subject_id_CBU070424/ROI_IFO_LH_fdt_FA/dtifit__FA_maths.nii'

img = nib.load(img_file)
img_dat = img.get_data()
s = img_dat.shape
img_dat_reshaped = img_dat.reshape(1,s[0]*s[1]*s[2])
img_dat_reshaped_flat = img_dat_reshaped.flatten()
fig = plt.figure()
ax = fig.add_subplot(111)
	
n,bins,patches = ax.hist(img_dat_reshaped_flat, 50, normed=1, facecolor='green', alpha=0.75)
	
# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
y = mlab.normpdf( bincenters, mu, sigma)
l = ax.plot(bincenters, y, 'r--', linewidth=1)

x.set_xlabel('Smarts')
ax.set_ylabel('Probability')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
ax.set_xlim(40, 160)
ax.set_ylim(0, 0.03)
ax.grid(True)

plt.show()
if __name__=='main':
	plt.show()

"""



def get_average(in_file1, in_file2):
    """
    Add doc
    """
    import nibabel as nib
    import numpy as np
    from nipype.interfaces.base import traits
    import os
    img1 = nib.load(in_file1)
    img2 = nib.load(in_file2)
    average_data = np.divide(img1.get_data()+img2.get_data(),2)
    average_img = nib.Nifti1Image(average_data,img1.get_affine())
    outfilename = os.path.abspath('average_image.nii')
    nib.save(average_img, outfilename)
    #return traits.File(outfilename)
    return outfilename


"""
Ziegler's helper functions (see e.g. camino_dti_tutorial.py) - 
putting them here to un-clutter some analysis scripts...

Define some helper functions that identify voxel and data dimensions of input images
"""

def get_vox_dims(volume):
    """
    Usage:
            [dim1, dim2, dim3] = get_vox_dims(volume)
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
    Usage:
            [dim1, dim2, dim3] = get_data_dims(volume)
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
    Usage:
            [dim1, dim2, dim3] = get_data_dims(volume)
    """

    import nibabel as nb
    nii = nb.load(volume)
    return nii.get_affine()

def select_aparc(list_of_files):
    for in_file in list_of_files:
        if 'aparc+aseg.mgz' in in_file:
            idx = list_of_files.index(in_file)
    return list_of_files[idx]

def select_aparc_annot(list_of_files):
    for in_file in list_of_files:
        if '.aparc.annot' in in_file:
            idx = list_of_files.index(in_file)
    return list_of_files[idx]




"""
temp change made here
"""


