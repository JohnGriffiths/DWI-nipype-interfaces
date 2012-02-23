"""
Nipype interfaces for various functions from 'dipy' (diffusion imaging in python)

see 
http://nipy.sourceforge.net/dipy/examples_built/nii_2_tracks.html
"""

import numpy as np
import nibabel as nib
import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti
from dipy.tracking.propagation import EuDX

from IPython import embed
"""
Isotropic voxel sizes required
dipy requires its datasets to have isotropic voxel size. If you have datasets with anisotropic voxel size then you need to resample with isotropic voxel size. We have provided an algorithm for this. You can have a look at the example resample_aniso_2_iso.py
"""
from dipy.align.aniso2iso import resample

def resample_aniso2iso(img_file, newdims=(3., 3., 3), filename=None):
	"""
	resample_aniso2iso(img, newdims=(3., 3., 3), filename=None)
	
	image can be either a nibabel image object or a filename (string)
	
	if filename is specified, saves new image as filename
	otherwise it is returned to the workspace 
	
	newdims can either be a TUPLE of dimensions or an image 
	file (either STRING for a filename to read in, or a 
	nibabel.nifti1.Nifti1Image object)
	"""
	if type(img_file)==str:
		print 'loading in diffusion data...'
		img=nib.load(img_file)
		data=img.get_data()
	elif str(type(img_file)) == "<class 'nibabel.nifti1.Nifti1Image'>":
		# (clumsy; temporary fix)
		img = img_file
		data = img.get_data()
	affine=img.get_affine()
	zooms=img.get_header().get_zooms()[:3]
	#zooms
	
	if type(newdims)==tuple:	
		new_zooms=newdims
	elif type(newdims)==str:
		new_zooms_img = nib.load(newdims)
		new_zooms = new_zooms_img.get_header().get_zooms()[:3]
	elif str(type(newdims)) == "<class 'nibabel.nifti1.Nifti1Image'>":
		# (clumsy; temporary fix)
		new_zooms = newdims.get_header().get_zooms()[:3]

	data2,affine2=resample(data,affine,zooms,new_zooms)
	#data2.shape
	
	img2=nib.Nifti1Image(data2,affine2)

	if filename==None:
		return img2
	else:
		nib.save(img2,filename)


def tenfit(data, bvals, bvecs, thresh=50):
	"""
	bvals and bvecs are text files, as per the fsl format
	data is a 4D nifti image
	
	thresh is (?)
	"""
	print 'fitting diffusion tensor...'
	if type(data) == str:
		print 'loading in diffusion data...'	
		img = nib.load(data)
	else:	
		img = data
	
	img_dat = img.get_data()
	
	if type(bvals) == str:
		print 'loading in bvals file'
		bvals=np.loadtxt(bvals)
		
	if type(bvecs) == str:
		print 'loading in bvecs file'
		gradients=np.loadtxt(bvecs).T
	
	print 'fitting tensor...'
	ten = dti.Tensor(img_dat, bvals, gradients, thresh)
	return ten

def get_FA_img(dat_img, ten=None,filename=None, bvals=None,bvecs=None):
	"""
	if filename is given, saves image to filename
	otherwise returns image
	
	if 'ten' is supplied the fa is calculated 
	directly from there (quicker); otherwise
	tensor is calculated from the image file
	"""
	if type(dat_img)==str:
		print 'loading in diffusion data...'
		dat_img = nib.load(dat_img)
		
	if ten==None:
		ten = tenfit(dat_img,bvals,bvecs)
	
	fa = ten.fa()
	fa_img = nib.Nifti1Image(fa,dat_img.get_affine())
	if filename==None:
		return fa_img
	else:
		nib.save(fa_img,filename)

	



"""
Calculate FA
- wls option
"""


"""
Reslice
"""


