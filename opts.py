#Initialize pointers
basenet = 'weights/vgg16_reducedfc.pth'
data_set = 'VOC'
dataset_root = '//datasets/ee285f-public/PascalVOC2012/'
save_folder = 'trained_weights/'
trained_model = 'weights/ssd300_mAP_77.43_v2.pth'
eval_save_folder = 'eval/'
devkit_path = 'devkit_path/'
output_dir = "out/"

#Run related metaparameters

batch_size = 32
resume = None

#Optimization metaparameters
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
    
confidence_threshold = 0.01
top_k = 5
cuda = True
cleanup = True

YEAR = '2012'
dataset_mean = (104, 117, 123)
set_type = 'val'