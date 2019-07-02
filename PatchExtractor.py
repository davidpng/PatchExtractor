#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:05:40 2019

@author: David P. Ng, MD
@email: david.ng@utah.edu

"""
import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import h5py
import datetime

def extract_patches(image, mask=None, patch_size=None, overlap = 0.5, mask_coverage = 0.75):
    '''
    David P. Ng, MD
    david.ng@utah.edu
    
    image - ndarray (n x m x 3) containing image data
    mask - ndarray (n x m) containing 0s and 1s
    patch_size - int or tuple that describes the size in pixels of the patch desired
    overlap - float32 between 0 and 1 that specifise the % of overlap with neighboring patches
    mask_coverage - float32 between 0 and 1 that specifies that amount of coverage of the given mask in
                    order to accept the patch in question
    '''
    import numpy as np
    #initialize mask
    if mask is None:
        mask = np.ones(image.shape)
        
    #initialize patch_size
    if patch_size is None:
        #if no patch size is given, patch will be the image
        patch_size = image.shape[0:2]
    elif isinstance(patch_size,int):
        # if patch_size is an int, make it a square tuple
        patch_size = (patch_size,patch_size) 
    elif not isinstance(patch_size,tuple):
        raise TypeError("patch_size must be an int or tuple")
    
    #check mask_coverage is between zero and one
    if 0 < mask_coverage < 1:
        pass
    else:
        raise ValueError("mask_coverage must be between 0 and 1 (0,1)")
        
    #check overlap is between zero and one
    if 0 < overlap < 1:
        pass
    else:
        raise ValueError("overlap must be between 0 and 1 (0,1)")
               
    mask_max = patch_size[0]*patch_size[1] #create a maximium mask value
    patch_list=[]
    mask_list=[]
    row_idx=0
    col_idx=0
    rejected_patches = 0
    num_patches = 0
    while row_idx < image.shape[0] - patch_size[0]:

        while col_idx < image.shape[1] - patch_size[1]:
            #create the region
            region = (slice(row_idx, row_idx + patch_size[0]),
                      slice(col_idx, col_idx + patch_size[1]))
            col_idx += int(round(patch_size[1]*(1-overlap)))
            num_patches += 1
            
            if mask[region].sum()/mask_max > mask_coverage:
                #if there is sufficent mask coverage by this selected patch, then append this
                #patch to the patch list
                patch = image[region]
                patch_list.append(patch)
            else:
                rejected_patches += 1
                
        #iterate the row index after columns are done
        row_idx += int(round(patch_size[0]*(1-overlap)))
        #reset column index
        col_idx=0
        
    #to see how many patches were rejected on the basis of the mask
    #print(rejected_patches/num_patches)
    return np.dstack(patch_list)
    

def input_arguments():
    """
    set up input arguments
    """
    parser = argparse.ArgumentParser(description='Takes images and extracts patches based on the associated mask')

    parser.add_argument('-d','--directory',
                    action='store', 
                    type=str,
                    nargs='?',
                    dest='directory',
                    help='Directory with image files to patch')

    parser.add_argument('-f','--filename',
                    action='store', 
                    type=str,
                    nargs='?',
                    dest='filename',
                    help='Get filename of files to patch')

    parser.add_argument('-o','--output',
                    action='store',
                    type=str,
                    nargs='?',
                    dest='output',
                    help='create filename for output, defaults to Patches [TIME].hdf5')

    parser.add_argument('--overlap',
                    action='store',
                    type=float,
                    nargs='?',
                    default=0.25,
                    dest='overlap',
                    help='specify the amount of overlap between patches')

    parser.add_argument('--mask_coverage',
                    action='store',
                    type=float,
                    nargs='?',
                    default=0.75,
                    dest='mask_coverage',
                    help='specifies that amount of coverage of the given mask in\
                    order to accept the patch in question')                
                    
    parser.add_argument('--patch_size',
                    action='store',
                    type=int,
                    nargs='+',
                    dest='patch_size',
                    help='specifies the patch size either as a tuple or int')                
                    
    parser.add_argument('-v','--verbose',
                    action='store_true',
                    dest='verbose',
                    help='output flags on completion of various tags')  
        
    parser.add_argument('--version', action='version', version='1.0')

    return parser.parse_args()

    
class get_image_masks:
    """
    Class for obtaining file or files with associated masks
    """
    def __init__(self, results):
        self.image_mask_names = None
        self.verbose = results.verbose
        
        if self.verbose:
            print(results)
            
        #set up name of hdf5 file
        if results.output is None:
            #default is Patches with datetime
            now = datetime.datetime.now()
            now = now.strftime('%Yy%mm%dd %Hh%Mm%Ss')
            self.hdf5_filename = "Patches {}.hdf5".format(now)
        else:
            self.hdf5_filename = results.output
        
        #check arguments
        if results.directory is not None:
            self.directory = results.directory
            self.image_mask_names = self.scrape_directory(self.directory)
        elif results.filename is not None:
            #if there is something in filename, grab that file and mask
            self.image_mask_names = self.grab_file(results.filename)
        else:
            #make the dir the pwd and scrape the pwd
            self.directory = os.getcwd()
            self.image_mask_names = self.scrape_directory(self.directory)
        #load images files and mask into a dataframe
        self.image_masks_df = self.load_files()
        #handle patches
        if results.patch_size is None:
            raise ValueError("Patch size is not given")
        elif len(results.patch_size) == 1: # single input
            self.patch_size = (results.patch_size[0],results.patch_size[0])
        elif len(results.patch_size) == 2: # two inputs
            self.patch_size = (results.patch_size[0],results.patch_size[1])
        elif len(results.patch_size) > 2: # too many inputs
            raise ValueError("Too many parameters in patch_size")
        #handle overlap
        self.overlap = results.overlap
        
        #handle mask coverage
        self.mask_coverage = results.mask_coverage
        
        #meat of this operation
        self.process_patches()
        #save patches to HDF5 file
        if self.verbose:
            print("Patches saved to {}".format(self.hdf5_filename))
        
    def load_files(self):
        # loads files into a dataframe
        output = []
        for index,rows in self.image_mask_names.iterrows():
            image = plt.imread(rows['imagename'])
            mask = plt.imread(rows['maskname'])
            filename = os.path.basename(rows['imagename'])
            row_dic = {"filename":filename, "image":image, "mask":mask}
            output.append(row_dic)
        
        return pd.DataFrame(output)
    
    def process_patches(self):
        """
        Due to memory concerns, this will process and place patches in an hdf5 file
        """
        #output_list=[]
        for index,rows in self.image_masks_df.iterrows():
            rows['filename']
            patch_list = extract_patches(image = rows['image'],
                                         mask = rows['mask'],
                                         patch_size=self.patch_size,
                                         overlap = self.overlap,
                                         mask_coverage = self.mask_coverage)
            row_dict = {"filename":rows['filename'],"patches":patch_list}
            if self.verbose:
                print("patch list is of size {}".format(patch_list.shape))
            #run save patches to HDF5 file    
            self._save_patches(row_dict,self.hdf5_filename)

    
    def _save_patches(self,row_dict,filename="patches_database.hdf5"):
        """
        Save patches to a hdf5 file with keys based on file names, and patches stacked along
        the last (color) dimension.  
        N.B. - Iterate 3D array along 3rd axis with step size 3  
        """
        output_path= os.path.join(self.directory,filename)
        with h5py.File(output_path,'a') as f:
            f.create_dataset(row_dict['filename'],
                             data=row_dict['patches'],
                             chunks=True,
                             compression="gzip")
        if self.verbose:
            print("patches for file {} written to file".format(row_dict['filename']))
        
    def scrape_directory(self,directory):
        """
        Returns a dataframe of images and masks
        """

        file_list=[]
        mask_list=[]
        
        for file in os.listdir(directory):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tif"):
                if 'mask.' in file:
                    mask_list.append(file)
                else:
                    file_list.append(file)
        #check file_list against mask_list
        image_mask_list = []
        for image in file_list:
            for mask in mask_list:
                if os.path.splitext(image)[0] in os.path.splitext(mask)[0]:
                    
                    image_mask_pair = {"imagename":os.path.join(directory,image),
                                       "maskname":os.path.join(directory,mask)}
                    image_mask_list.append(image_mask_pair)
                    break

                
        if not image_mask_list:
            raise ValueError("No image mask pairs were found")
            
        return pd.DataFrame(image_mask_list)
    
    def grab_file(self,filepath):
        mask = None
        working_dir = os.path.dirname(os.path.abspath(filepath))
        if not os.path.isfile(filepath):
            raise ValueError("{} does not exist".format(filepath))
        else:
            #get all files in directory with filename
            for file in os.listdir(working_dir):
                #check to see if potential mask names match the file name
                if os.path.basename(filepath)[0] in os.path.splitext(file)[0]:
                        #check if marked as mask
                        if 'mask.' in file:
                            # return the aboslute path of the mask from the working dir
                            mask = os.path.join(working_dir,file)
                            break
        if mask is None:
            #error check for now
            raise ValueError("Mask for {} does not exist".format(filepath))
            pass
        #make a list with one dictionary to make pandas happy
        image_mask_pair = [{"imagename":filepath,"maskname":mask}]
        return pd.DataFrame(image_mask_pair)
                        
                        
        

def main():
    results=input_arguments()
    filenames = get_image_masks(results)
    
if __name__== "__main__":
    main()


