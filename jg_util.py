#-------------------------------------------------------------------------------
# Name:        jg_util.py
# Purpose:     Collection of python functions for general usage (see also Barry's ones in 'bd_util.py')
# Author:      Joh
# Created:     21/02/2011
# Copyright:   (c) CSLB 2011
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import os
import xlrd
import xlwt
import xlutils
import glob
import numpy
import jg_util
import sys
import gzip
import string

def not_in_list(list1,list2):
	# got this info from http://python.about.com/od/pythonstandardlibrary/a/set_types.htm
	l1_set = set(list1)
	l2_set = set(list2)
	list1_not_in_list2 = l1_set.difference(l2_set)
	list2_not_in_list1 = l2_set.difference(l1_set)
	return list1_not_in_list2, list2_not_in_list1
	
	
def find_in_subdirectory(filename, subdirectory=''):
	"""
	Walks subdirectories to find a file and returns.
	Default start location is the current working directory.
	Optionally, a different directory can be set as the search's starting location
	(modified from http://code.activestate.com/recipes/577027-find-file-in-subdirectory/; removed the 'raise' bit)	
		
	Usage: 
			dirs = jg_util.find_in_subdirectory(filename,subdirectory)
	"""
	if subdirectory:
		path = subdirectory
	else:
        	path = os.getcwd()
	
	found_file=0
	matches = []
	for root, dirs, names in os.walk(path):
		if filename in names:
			matches.append(os.path.join(root, filename))
			found_file=1
		#else: raise 'File not found'
	if found_file == 0: print 'File not found' 
	else: print 'Found file'
	return matches

def find_in_subdirectory_string(string, subdirectory=''):
	"""
	[JG modification of 'find_in_subdirectory' - also 
	searches for files that include the supplied string]
	
	Walks subdirectories to find a file and returns.
	Default start location is the current working directory.
	Optionally, a different directory can be set as the search's starting location
	(modified from http://code.activestate.com/recipes/577027-find-file-in-subdirectory/; removed the 'raise' bit)	
		
	Usage: 
			dirs = jg_util.find_in_subdirectory_string(string,subdirectory)
	"""
	if subdirectory:
		path = subdirectory
	else:
        	path = os.getcwd()
	
	found_file=0
	matches = []
	for root, dirs, names in os.walk(path):
		for filename in names:
			if string in filename:
				matches.append(os.path.join(root, filename))
				found_file=1
				#else: raise 'File not found'
	if found_file == 0: print 'File not found' 
	else: print 'Found file'
	return matches


def spreadsheet2dict_keysfromfirstcol(xl_filename, sheetname):
	# IDEA: CHANGE 'SPREADSHEET2DICT' TO A CLASS AND INCORPORATE ALL OF THE VERSIONS BELOW
	# creates a dictionary, d, from an excel spreadsheet - where keys come from the first column
	# example usage: d =jg_util. spreadsheet2dict_keysfromfirstcol('xl_filename','sheetname')
	wb = xlrd.open_workbook(xl_filename)
	ws = wb.sheet_by_name(sheetname)
	d = {}
	for r in range(1,ws.nrows):
		d[ws.cell_value(r,0)] = ws.row_values(r)
	return d

def spreadsheet2dict_keysfromfirstrow(xl_filename, sheetname):
	# creates a dictionary, d, from an excel spreadsheet - where keys come from the first row
	# example usage: d =jg_util.spreadsheet2dict_keysfromfirstrow('xl_filename','sheetname')
	wb = xlrd.open_workbook(xl_filename)
	ws = wb.sheet_by_name(sheetname)
	d = {}
	for c in range(0,ws.ncols):
		d[ws.cell_value(0,c)] = ws.col_values(c)
	return d

def spreadsheet2dict_keysfromfirstrowandfirstcol(xl_filename, sheetname):
	# creates a dictionary, d, from an excel spreadsheet - where keys are pairs of [items?]
	# example usage: d =jg_util.spreadsheet2dict_keysfromfirstrowandfirstcol('xl_filename','sheetname')
	wb = xlrd.open_workbook(xl_filename)
	ws = wb.sheet_by_name(sheetname)
	d = {}
	for r in range(1,ws.nrows):
		for c in range(1,ws.ncols):
			d[(ws.cell_value(r,0), ws.cell_value(0,c))] = ws.cell_value(r,c)
	return d

def spreadsheet2dict_keysfromfirstrowandfirstcol_2D(xl_filename, sheetname):
	# creates a 2D dictionary, d, from an excel spreadsheet, with two separate keys for each value
	# example usage: d =jg_util.spreadsheet2dict_keysfromfirstrowandfirstcol_2D('xl_filename','sheetname')
	wb = xlrd.open_workbook(xl_filename)
	ws = wb.sheet_by_name(sheetname)
	d = {}
	for r in range(1,ws.nrows):
		for c in range(1,ws.ncols):
			d[ws.cell_value(r,0),ws.cell_value(0,c)] = ws.cell_value(r,c)
	return d

def make_CSLid_subject_data_dicts(xl_filename, sheetname, value_list):
	# matches entries from any list of database info values (value_list) with the corresponding CSLid, 
	# identified from an excel file (copied+pasted from the database), with CSLids in the 1st column 
	# Outputs four dictionaries:
	# 1. 'CSLid_all_data_dict':		key = CSLid, values = all database values
	# 2. 'CSLid_value_list_dict': 		key = CSLid, values = value_list items
	# 3. 'value_list_all_data_dict':	key = value_list items, values = all database values
	# 4. 'value_list_CSLid_dict': 		key = value_list_items, values = all database values

	import jg_util
	import os

	if not os.path.isfile(xl_filename):
		raise NameError, "can't find excel file %s " % xl_filename
	CSLid_all_data_dict_orig = jg_util.spreadsheet2dict_keysfromfirstcol(xl_filename,sheetname)
	CSLid_all_data_dict = {}
	CSLid_value_list_dict = {}
	value_list_all_data_dict = {}
	value_list_CSLid_dict = {}
	values_not_matched = []
	for vl in value_list:
		match = [k for k,v in CSLid_all_data_dict_orig.iteritems() if vl in v] # find the CSLid corresponding to the current value vl in the value list
		match_omit1stdigit = [k for k,v in CSLid_all_data_dict_orig.iteritems() if vl[1:] in v] # find 
		if not(len(match)==0) or not(len(match_omit1stdigit)==0): # if a CSLid has been found for that number...
			if not(len(match)==0):
				match_to_use = match
			elif not(len(match_omit1stdigit)==0):
				match_to_use = match_omit1stdigit
			# add the CSLid to the dictionaries with value list items as keys
			value_list_all_data_dict[vl] = CSLid_all_data_dict_orig[match_to_use[0]]
			value_list_CSLid_dict[vl] = match_to_use[0]
			# ...and add the data for that CSLid
			CSLid_all_data_dict[match_to_use[0]] = CSLid_all_data_dict_orig[match_to_use[0]]
			if match_to_use[0] in CSLid_value_list_dict.keys(): 
				# if the CSLid has already been put in this dictionary, then append it to the existing entry, which is a list of (e.g.) CBU numbers corresponding to that CSLid 
				vl_temp = CSLid_value_list_dict[match_to_use[0]]
				vl_temp.append(vl)
				CSLid_value_list_dict[match_to_use[0]] = vl_temp
			else:
				CSLid_value_list_dict[match_to_use[0]] = [vl]
		else: # if a CSLid has not been found for the current value vl
			values_not_matched.append(vl)			
	return CSLid_all_data_dict, CSLid_value_list_dict, value_list_all_data_dict, value_list_CSLid_dict, values_not_matched



	# Also from this webpage:
	
		#I initially wrote findInSubdirectory to make it more convenient to pass in data files to python scripts from the command line. Specifically,
		
		## Create parser for command line options
		#parser = OptionParser()
		
		## Get filename/resolve path
		#parser.add_option("-f", "--file", dest="filename", type="string",
				#help="Specifies log file", metavar="FILE")
		
		#parser.add_option("-s", "--subdirectory", dest="subdirectory", type="string",
				#help="Specifies path to logfile", metavar="subdirectory")
		
		#subdirectory = options.subdirectory
		#filename = options.filename
		
		#path_to_file = findInSubdirectory(filename, subdirectory)
		#logData = open(path_to_file, 'r')


def locate(pattern, root=os.curdir):
# stolen from http://code.activestate.com/recipes/499305/
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    import os, fnmatch
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)
	#"os.walk is a very nice replacement for os.path.walk, which I never did feel comfortable with. There's one very common pattern of usage, though, which still benefits from a simple helper-function; locating all files matching a given file-name pattern within a directory tree."
	
	# Also from this website:
	
	#An example - I needed to identify all malformed XML files within my project. With the help of locate (and ElementTree) all I needed was
	#from xml.parsers.expat import ExpatError
	#import xml.etree.ElementTree as ElementTree
	
	#for xml in locate("*.xml"):
	#try:
		#ElementTree.parse(xml)
	#except (SyntaxError, ExpatError):
		#print xml, "\tBADLY FORMED!"
	

def get_scan_nums_from_database_doc(xl_filename, sheetname):
	scan_num_cols = ['WBIC no - current','WBIC no - previous 1','CBU scan no - current','CBU scan no - previous 1','CBU scan no - previous 2','CBU scan no -  previous 3','CBU scan no -  previous 4','CBU scan no -  previous 5']
	d1 = jg_util.spreadsheet2dict_keysfromfirstcol(xl_filename, sheetname)
	d2 = jg_util.spreadsheet2dict_keysfromfirstrowandfirstcol(xl_filename, sheetname)
	scan_nums_list = []
	for r in d1.iterkeys():
		for c in scan_num_cols:
			if len(d2[r,c])>0:
				scan_nums_list.append(d2[r,c])
	return scan_nums_list


def find_scan_date_and_age_at_scan_from_CBUnum(xl_filename, sheetname, CBUnum):

	from datetime import date, datetime, time
	from xlrd import open_workbook, xldate_as_tuple

	import jg_util
	import xlrd
	scan_num_scan_date_dict = {'WBIC no - current':'WBIC no - date of current','WBIC no - previous 1':'WBIC no - date of previous 1','CBU scan no - current':'CBU scan no - date of current', 'CBU scan no - previous 1':'CBU scan no - date of previous 1', 'CBU scan no - previous 2':'CBU scan no - date of previous 2','CBU scan no -  previous 3':'CBU scan no - date of previous 3','CBU scan no -  previous 4':'CBU scan no - date of previous 4', 'CBU scan no -  previous 5':'CBU scan no - date of previous 5'}

	[CSLid_all_data_dict, CSLid_CBUnums_dict, CBUnums_all_data_dict, CBUnums_CSLid_dict] = jg_util.make_CSLid_subject_data_dicts(xl_filename, sheetname, CBUnum)
	d1 = jg_util.spreadsheet2dict_keysfromfirstcol(xl_filename, sheetname)
	d2 = jg_util.spreadsheet2dict_keysfromfirstrowandfirstcol(xl_filename, sheetname)
	
	b = xlrd.open_workbook(xl_filename)
	
	for r in d1.iterkeys():
		for c in scan_num_scan_date_dict.iterkeys():
			if d2[r,c] == CBUnum:
				scan_date = xldate_as_tuple(d2[r,scan_num_scan_date_dict[c]],b.datemode)
				DOB = xldate_as_tuple(d2[r, 'DOB'],b.datemode)
				age_at_scan = jg_util.datediff(DOB,scan_date,'Tuple')
	if len(scan_date)==0:
		scan_date.append('couldn''t find scan number')	
	return scan_date, DOB, age_at_scan



def convert_date_format(date,conversion_type):
	
	import datetime as dt
	from xlrd import open_workbook, xldate_as_tuple

	if conversion_type == 'British to American':
		# day/month/year to month/day/year
		d1, m1, y1 = (str(x) for x in date.split('/'))
		new_date = [m1+'/'+d1+'/'+y1]	
	
	if conversion_type == 'Tuple to American':
		new_date = [str(date[1])+'/'+str(date[2])+'/'+str(date[0])]			
	
	if conversion_type == 'British to Tuple':
		d1, m1, y1 = (str(x) for x in date.split('/'))
		# not done this yet
	
	if conversion_type == 'American to Tuple':
		m1, d1, y1 = (str(x) for x in date.split('/'))
		# not done this yet
	
	if conversion_type == 'American to British':
		# month/day/year to day/month/year
		# [ note: has the same effect as British to American i.e. swapping round the first two entries]
		m1, d1, y1 = (str(x) for x in date.split('/'))
		new_date = [d1+'/'+m1+'/'+y1]	
	
	if conversion_type == 'Tuple to British':
		new_date = [str(date[2])+'/'+str(date[1])+'/'+str(date[0])]	
	 	
	return new_date[0]
	
	
def datediff(date1,date2,date_format):
	# stolen and modified from http://www.daniweb.com/code/snippet216974.html
	# find the days between 2 given dates
	# tested with Python25 HAB	
	# US format month/day/year

	import datetime as dt
	from xlrd import open_workbook, xldate_as_tuple

	if date_format == 'British':
		date1 = jg_util.convert_date_format(date1,'British to American')
		date2 = jg_util.convert_date_format(date2,'British to American')
	elif date_format == 'Tuple':
		date1 = jg_util.convert_date_format(date1,'Tuple to American')
		date2 = jg_util.convert_date_format(date2,'Tuple to American')
		
	m1, d1, y1 = (int(x) for x in date1.split('/'))
	m2, d2, y2 = (int(x) for x in date2.split('/'))
	
	new_date1 = dt.date(y1, m1, d1)
	new_date2 = dt.date(y2, m2, d2)
	
	datediff = new_date2 - new_date1
	datediff_days = datediff.days
	datediff_years = datediff_days/365
	print 'Difference in days = %d' % datediff_days  # 291
#	print 'Difference in years = %d' % datediff_years  # 291
	print 'Difference in years= ~%d' % datediff_days.days/360  # 291
	
	return datediff_years
	
	

def lookup_value_from_database_doc(xl_filename, sheetname,CSLid,value):
	d2 = jg_util.spreadsheet2dict_keysfromfirstrowandfirstcol(xl_filename, sheetname)
	output = d2[CSLid, value]
	return output


def find_text_string_in_file(filename, text_string):
	# stolen mostly from http://www.daniweb.com/forums/thread113927.html
	# Example usage: t = jg_util.find_text_string_in_file(filename, text_string)
	# (note: don't do [t] = jg_util.....() - or you get a 'Value Error: too many values to unpack' msg
	file = open(filename, "r")
	file_text = file.read()
	file.close()
	text_at_indices = []
	found_text = file_text.find(text_string)
	#text_at_indices.append(found_text)
	while found_text >0:
		print 'string "',text_string,'" found at index     ',found_text
		text_at_indices.append(found_text)
		found_text = file_text.find(text_string, found_text+1)
	return text_at_indices # couldn't get this to work.

def gunzip(file):
	"""Gunzips the given file and then removes the file.
	(stolen from www.researchut.com/blog/archive/2006/10/17)"""
	r_file = gzip.GzipFile(file, 'r')
	write_file=string.rstrip(file, '.gz')
	w_file = open(write_file, 'w')
	w_file.write(r_file.read())
	w_file.close()
	r_file.close()
	os.unlink(file)
	sys.stdout.write("%s gunzipped.\n" % (file))
	
""" 
'jg_list_DTI_dirs_in_raw_data.py'
This file defines a function 'list_dirs' that will search through a top-level directory for lower level folders containing the substring 'DTI', which the directories containing DTI scans from a session always do. It sorts the results of this search by which scan sequence used (3 options: CBU_DTI_64D_1A, CBU_DTI_64D_2A, and CBU_DTI_64InLea.
	
	Usage: 		
	 			ipython		
				import jg_list_DTI_dirs_in_raw_data
				all_lists = jg_list_DTI_dirs_in_raw_data.list_dirs()
	
	This will oute above lines will output a large dictionary, 'all_lists', into the workspace, which contains some useful and some less useful lists. See the full list of lists contained in all_lists with 
	 			
				all_lists.keys()
	  
	   The most important and useful one is the list of unique CBU numbers that were found to have DTI data. Access this with
	 			all_lists['all_DTI_CBUnums_unique']
	     
	Other useful ones are 
	 			all_lists['all_64D_1A_CBUnums_unique']
				all_lists['all_64D_2A_CBUnums_unique']
		 		all_lists['all_64InLea_CBUnums_unique']
"""

def list_dirs(top_level_dir):
	"""
	This file defines a function 'list_dirs' that will search through a top-level directory for lower level folders containing the substring 'DTI', which the directories containing DTI scans from a session always do. It sorts the results of this search by which scan sequence used (3 options: CBU_DTI_64D_1A, CBU_DTI_64D_2A, and CBU_DTI_64InLea.
		
		Usage: 		
					ipython		
					import jg_list_DTI_dirs_in_raw_data
					all_lists = jg_list_DTI_dirs_in_raw_data.list_dirs()
		
		This will oute above lines will output a large dictionary, 'all_lists', into the workspace, which contains some useful and some less useful lists. See the full list of lists contained in all_lists with 
					
					all_lists.keys()
		
		The most important and useful one is the list of unique CBU numbers that were found to have DTI data. Access this with
					all_lists['all_DTI_CBUnums_unique']
		
		Other useful ones are 
					all_lists['all_64D_1A_CBUnums_unique']
					all_lists['all_64D_2A_CBUnums_unique']
					all_lists['all_64InLea_CBUnums_unique']
	"""
		
	import os
	import re
		
	# Define the variables
	raw_data_dir = top_level_dir
	a = {} # all lists
	all_list_names = ['all_DTI_dirs', 'all_DTI_CBUnums','all_DTI_CBUnums_unique', 'all_64D_1A_CBUnums', 'all_64D_1A_CBUnums_unique', 'all_64D_2A_CBUnums', 'all_64D_2A_CBUnums_unique','all_64InLea_CBUnums', 'all_64InLea_CBUnums_unique','all_other_CBUnums','all_other_CBUnums_unique','all_DTI_nonCBU_dirs','all_DTI_nonCBU_dirs_unique', 'all_DTI_dirs_v2', 'all_fullpathnames']
	for l in all_list_names:
		a[l] = []
		
	# Now the loops
	for path, subdirs, files in os.walk(raw_data_dir):
		for name in files:
			fullpathname = os.path.join(path,name)
			a['all_fullpathnames'].append(fullpathname)	
	for fpn in a['all_fullpathnames']:
		re_search_CBU = re.search('CBU\d\d\d\d\d\d', fpn)
		re_search_DTI = re.search('DTI', fpn)
		re_search_64D_1A = re.search('64D_1A', fpn)
		re_search_64D_2A = re.search('64D_2A', fpn)
		re_search_64InLea = re.search('InLea', fpn)
		if not(re_search_DTI==None):
			a['all_DTI_dirs_v2'].append(fpn)
		if not(re_search_DTI==None) and not(re_search_CBU==None):
			a['all_DTI_dirs'].append(fullpathname)
			a['all_DTI_CBUnums'].append(re_search_CBU.group(0))
			if not(re_search_64D_1A==None):
				a['all_64D_1A_CBUnums'].append(re_search_CBU.group(0))
			if not(re_search_64D_2A==None):
				a['all_64D_2A_CBUnums'].append(re_search_CBU.group(0))
			if not(re_search_64InLea==None):
				a['all_64InLea_CBUnums'].append(re_search_CBU.group(0))
			if re_search_64D_1A==None and re_search_64D_2A==None and re_search_64InLea==None:
				a['all_other_CBUnums'].append(re_search_CBU.group(0))
	a['all_DTI_CBUnums_unique'] = set(a['all_DTI_CBUnums'])
	a['all_64D_1A_CBUnums_unique'] = set(a['all_64D_1A_CBUnums'])
	a['all_64D_2A_CBUnums_unique'] = set(a['all_64D_2A_CBUnums'])
	a['all_64InLea_CBUnums_unique'] = set(a['all_64InLea_CBUnums'])
	a['all_other_CBUnums_unique'] = set(a['all_other_CBUnums'])
	a['all_DTI_nonCBU_dirs_unique'] = set(a['all_DTI_nonCBU_dirs'])
	a['all_DTI_dirs_v2_unique'] = set(['all_DTI_dirs_v2'])
	return a
