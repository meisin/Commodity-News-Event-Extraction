SET TRAIN_FILE=dataset/event_extraction_train.json
SET TEST_FILE=dataset/event_extraction_test.json



python train.py --train_data_file=%TRAIN_FILE% --eval_data_file=%TEST_FILE% --batch_size=2 --transformer_type="combert"
