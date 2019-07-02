# PatchExtractor
python command line program to extract patches from images with a masked region

SYNTAX at command line

example

python PatchExtractor.py -d /TestFiles/ --patch_size 128 --overlap 0.2 -v

TestFiles contains images and masks for testing, generated with qupath

standard python dependencies: os, argparse, datetime

python dependencies to install: h5py, pandas, matplotlib
