# classifier_or_cfg_sample.py
SAMPLE_BATCH_SIZE = 4 # batchsize
SAMPLE_LOG_ROOT = '../logs_stage1' # Path of logs
SAMPLE_MODEL_PATH = '' # diffusion model checkpoint
SAMPLE_CLASSIFIER_PATH = '' # classifier model checkpoint
SAMPLE_CLASSIFIER_SCALE = 0.0 # classifier guided scale. The default value is 0.0, which means it is not used.
CFG = 0.0 # classifier free scale. The default value is 0.0, which means it is not used.
SAMPLE_DATASET_DIR = '../logs_stage1' # Path of generated data
SAMPLE_CATEGORY_NAME_LIST = ["0","1"] # List of image category names, len of the list is the num of categories.
SAMPLE_CATEGORY_NUM_LIST = [1,1]  # Number of images generated per category
SAMPLE_DATASET_NAME = 'dataname' # Name of generated data

# dual_bridge_sample.py
DUAL_SAMPLE_LOG_ROOT = '../logs_stage2' # Path of logs
DUAL_SAVE_DIR = '../logs_stage2/dataname' # Path of generated data
DUAL_BATCH_SIZE =  2 # batchsize
DUAL_GUIDED_SCALE = 0.0 # guided scale. The default value is 0.0, which means it is not used.
WHICH_GUIDE = "Entropy" # Entropy/Margin
DUAL_MODEL_PATH = '' # diffusion model checkpoint
DUAL_CLASSIFIER_PATH = '' # classifier model checkpoint.
DUAL_SOURCE_LABEL = "0,1"  # List of image category names, len of the list is the num of categories.
DUAL_TARGET_LABEL = "0,1"  # Please keep consistent with DUAL_SOURCE_LABEL
DUAL_VAL_DIR = "../logs_stage1/dataname" # Path to first stage data