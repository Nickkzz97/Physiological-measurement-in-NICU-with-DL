MODEL='physnet'
CHECKPOINT='wbcheckpointnicupatwise_new/best_model.pt'
OUT_DIR='outputppg_pearson_new/test_all'    
BATCH_SIZE=1
DEVICE='cuda:0'
TEST_CSV_PATH='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/phynetNegpea/extra.txt'
DATA_DIR='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/Datafolder/nicucrop/'
PPG_PATH='/home/nicky/nickynicu/Video-Based-Physiological-Measurement/deeplearning/ConventionalMethod/Whitebalance/conventionalResult/nicucrop'
Dataset='NICU'

python test.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE}  --data-path ${DATA_DIR}  --ppg_dir ${PPG_PATH}  --valid-csv-path  ${TEST_CSV_PATH}  --dataset ${Dataset}




