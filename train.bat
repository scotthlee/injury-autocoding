@echo off
set datadir=C:/data/niosh_ifund
set runlist=(run_1 run_2 run_3 run_4)

rem Preprocessing the text
python %cd%/src/preprocessing.py

rem Fine-tuning the BERT models--this will take a while!
for %%a in %runlist% do ^
python %cd%/bert/run_classifier.py ^
--task_name=cola ^
--do_train=true ^
--do_eval=false ^
--data_dir=%datadir%/ ^
--vocab_file=%datadir%/bert_models/uncased_base/vocab.txt ^
--bert_config_file=%datadir%/bert_models/uncased_base/bert_config.json ^
--init_checkpoint=%datadir%/bert_models/uncased_base/bert_model.ckpt ^
--max_seq_length=44 ^
--train_batch_size=32 ^
--learning_rate=2e-5 ^
--num_train_epochs=3 ^
--output_dir=%datadir%/train_runs/%%a/
