import nipype
import nipype.pipeline as pe
import nipype.interfaces.io as io
import nipype.interfaces.fsl as fsl

from bids.layout import BIDSLayout



input_folder='/scratch/akhanf/test_avg_t2/bids'

#this is where datasinks will go 
output_folder='/scratch/akhanf/test_avg_t2/out'

n_procs = 8


#this is the base_dir, where intermediate files and graphs go
work_folder= output_folder + '/nipype'



matching_string='sub-{subject}/anat/sub-{subject}*_T2w.nii.gz'


# get list of subjects from pybids
layout = BIDSLayout(input_folder)
subject_list = layout.get_subjects()


subj_iter = pe.Node(nipype.IdentityInterface(fields=['subject_id']),
                  name="subj_iter")
subj_iter.iterables = [('subject_id', subject_list)]


# Create SelectFiles node
templates={'in_images': matching_string}
sf = pe.Node(io.SelectFiles(templates, base_directory=input_folder),  name='selectfiles')


# can add some checks to make sure all images should actually be aligned and averaged (e.g. not different kinds of acquisitions)
# also add a check that there are at least 2 scans

#make a node that gets 1st image, to use as registration reference
def getFirst(in_files):
    out_file = in_files[0]
    return out_file

getRefImage = pe.Node(nipype.Function(function=getFirst, input_names=["in_files"], output_names=["out_file"]), name="getRefImage")

#make a node that gets every other image in the list
def getAllButFirst(in_files):
    out_files = in_files[1:]
    return out_files

getFloatingImages = pe.Node(nipype.Function(function=getAllButFirst,input_names=["in_files"],output_names=["out_files"]),name="getFloatingImages")

#flirt 6-dof registration, sinc interpolation
fsl_flirt = pe.MapNode(interface = fsl.FLIRT(dof=6,interp='sinc'), name='fsl_flirt',iterfield=['in_file'])

#merge reg'd images and ref image into one list again
aligned = pe.Node(nipype.Merge(2),name='aligned')

#merge images into 4d
fsl_merge = pe.Node(interface = fsl.Merge(dimension='t'), name='fsl_merge')

#average over 4d to create final image
fsl_avg = pe.Node(interface = fsl.MeanImage(), name='fsl_avg')


# Create DataSink object
sinker = pe.Node(nipype.DataSink(), name='sinker')
sinker.inputs.base_directory = output_folder


#Create a workflow to connect all those nodes
wf = nipype.Workflow('workflow')
wf.base_dir = work_folder


wf.connect(subj_iter,'subject_id',sf,'subject')
wf.connect(sf,'in_images',getRefImage,'in_files')
wf.connect(sf,'in_images',getFloatingImages,'in_files')

wf.connect(getRefImage,'out_file',fsl_flirt,'reference')
wf.connect(getFloatingImages,'out_files',fsl_flirt,'in_file')

wf.connect(getRefImage,'out_file',aligned,'in1')
wf.connect(fsl_flirt,'out_file',aligned,'in2')

wf.connect(aligned,'out',fsl_merge,'in_files')
    
wf.connect(fsl_merge,'merged_file',fsl_avg,'in_file')

wf.connect(fsl_avg,'out_file',sinker,'avgRuns')

# write the graph to file
wf.write_graph(graph2use='exec', format='png', simple_form=False) #make graph2use=flat if don't want to expand each subj

nipype.config.enable_debug_mode()

#Run the workflow
plugin = 'MultiProc' #adjust your desired plugin here
plugin_args = {'n_procs': n_procs} #adjust to your number of cores

wf.run(plugin=plugin, plugin_args=plugin_args )
