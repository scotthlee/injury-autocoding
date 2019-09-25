@echo off
set testfile=test.tsv
set datadir=C:/data/niosh_ifund
set runlist=(run_1 run_2 run_3 run_4)

rem Getting the predictions from the BERT models
for %%a in %runlist% do ^
python %cd%/bert/run_classifier.py ^
--task_name=cola ^
--do_predict=true ^
--data_dir=%datadir%/ ^
--vocab_file=%datadir%/bert_models/uncased_base/vocab.txt ^
--bert_config_file=%datadir%/bert_models/uncased_base/bert_config.json ^
--init_checkpoint=%datadir%/train_runs/%%a/model.ckpt-14433 ^
--max_seq_length=44 ^
--output_dir=%datadir%/test_runs/%%a/

rem Running the evaluation script
python %cd%/src/evaluation.py ^
--data_dir=%datadir%/ ^
--test_file=%testfile% ^
--mode=test ^
--text_only=True ^
--train_blender=False
