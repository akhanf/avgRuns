#!/usr/bin/env python

import argparse, os
import nipype
import nipype.pipeline as pe
import nipype.interfaces.io as io
import nipype.interfaces.fsl as fsl

from bids.layout import BIDSLayout


# argument parsing for app-like interface

parser = argparse.ArgumentParser(description='avgRuns BIDS app script.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant', 'group'])
parser.add_argument('--participant_label', help='The label(s) of the participant(s) '
                    'that should be analyzed. The label '
                    'corresponds to sub-<participant_label> from the BIDS spec '
                    '(so it does not include "sub-"). If this parameter is not '
                    'provided all subjects should be analyzed. Multiple '
                    'participants can be specified with a space separated list.',
                    nargs="+")

args = parser.parse_args()
bids_dir = args.bids_dir
out_dir = args.output_dir


# testing inputs:
# bids_dir = '../bids'
# out_dir = '../out_test2'



#make absolute paths
os.makedirs(out_dir, exist_ok=True)
bids_dir = os.path.abspath(bids_dir)
out_dir = os.path.abspath(out_dir)



# get list of subjects from pybids
layout = BIDSLayout(bids_dir)

# Extract sub_ids for SelectFiles
if args.participant_label:
    subject_list = args.participant_label
else:
    subject_list = layout.get_subjects()



n_procs = -1; # by default will estimate nprocs - use input argument to set this..

#this is the base_dir, where intermediate files and graphs go
work_folder= out_dir + '/nipype'



matching_string='sub-{subject}/anat/sub-{subject}*_T2w.nii.gz'



subj_iter = pe.Node(nipype.IdentityInterface(fields=['subject_id']),
                  name="subj_iter")
subj_iter.iterables = [('subject_id', subject_list)]


# Create SelectFiles node
templates={'in_images': matching_string}
sf = pe.Node(io.SelectFiles(templates, base_directory=bids_dir),  name='selectfiles')


# can add some checks to make sure all images should actually be aligned and averaged (e.g. not different kinds of acquisitions)
# also add a check that there are at least 2 scans

#make a node that gets 1st image, to use as registration reference
def get_first(in_files):
    out_file = in_files[0]
    return out_file

get_ref_image = pe.Node(nipype.Function(function=get_first, input_names=["in_files"], output_names=["out_file"]), name="get_ref_image")

#make a node that gets every other image in the list
def get_all_but_first(in_files):
    out_files = in_files[1:]
    return out_files

get_floating_images = pe.Node(nipype.Function(function=get_all_but_first,input_names=["in_files"],output_names=["out_files"]),name="get_floating_images")



#flirt 6-dof registration, sinc interpolation
fsl_flirt = pe.MapNode(interface = fsl.FLIRT(dof=6,interp='sinc'), name='fsl_flirt',iterfield=['in_file'])

#merge reg'd images and ref image into one list again
aligned = pe.Node(nipype.Merge(2),name='aligned')

#merge images into 4d
fsl_merge = pe.Node(interface = fsl.Merge(dimension='t'), name='fsl_merge')

#average over 4d to create final image
fsl_avg = pe.Node(interface = fsl.MeanImage(), name='fsl_avg')


# Create DataSink object
datasink = pe.Node(nipype.DataSink(), name='datasink')
datasink.inputs.base_directory = out_dir

## Use the following DataSink output substitutions
substitutions = [('_subject_id_', 'sub-'),
                 ('_T2w_merged_mean.nii.gz', '_proc-avg_T2w.nii.gz'),
                 ('_run-01_', '_')]

datasink.inputs.substitutions = substitutions

#Create a workflow to connect all those nodes
wf = nipype.Workflow('workflow')
wf.base_dir = work_folder


wf.connect(subj_iter,'subject_id',sf,'subject')

wf.connect(sf,'in_images',get_ref_image,'in_files')
wf.connect(sf,'in_images',get_floating_images,'in_files')

wf.connect(get_ref_image,'out_file',fsl_flirt,'reference')
wf.connect(get_floating_images,'out_files',fsl_flirt,'in_file')

wf.connect(get_ref_image,'out_file',aligned,'in1')
wf.connect(fsl_flirt,'out_file',aligned,'in2')

wf.connect(aligned,'out',fsl_merge,'in_files')
    
wf.connect(fsl_merge,'merged_file',fsl_avg,'in_file')

wf.connect(fsl_avg,'out_file',datasink,'avgRuns')



# write the graph to file
wf.write_graph(graph2use='colored', format='png', simple_form=True) #other options: graph2use=flat, graph2use=exec 


#Run the workflow
plugin = 'MultiProc' #adjust your desired plugin here
#plugin = 'Linear' # use this for debugging
#nipype.config.enable_debug_mode()

if (n_procs > 0 ):
    plugin_args = {'n_procs': n_procs} #adjust to your number of cores
else:
    plugin_args = {}

wf.run(plugin=plugin, plugin_args=plugin_args )
