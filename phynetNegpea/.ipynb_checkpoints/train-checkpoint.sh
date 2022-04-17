EXP_NAME='PhysnetTraining'
#DATA_DIR='data_folder/UBFC_DATASET/DATASET_2'
EXP_DIR='wbcheckpointnicupatwise_new'
TRAIN_CSV_PATH='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/phynetNegpea/final_trainrand.txt'
DEV_CSV_PATH='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/phynetNegpea/final_validrand.txt'
DATA_DIR='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/Datafolder/nicucrop/'
PPG_PATH='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/conventionalResult/nicucrop'


python train.py --exp_name ${EXP_NAME} --data_dir ${DATA_DIR} --ppg_dir ${PPG_PATH}  --train_txt ${TRAIN_CSV_PATH} --dev_txt ${DEV_CSV_PATH} --exp_dir ${EXP_DIR} 