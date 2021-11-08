### Dependencies
```
pip install -r requirements.txt
```
### Data
Required data files are already generated and located in `/home/ayeffkay/rubert_tiny`. Provided scripts can generate files with same names.
For training you'll need 
* folder `processed_binarized` (binarized shards)
* `rubert_tiny_weights.pth` (student weights for initialization)
* `ru_convers` (teacher with LM head and fixed configs), `distilrubert_tiny_cased_convers` (not trained student)
* `teacher2student.pickle` (mapping dict), `t2s_padded.pickle`, `s2t_padded.pickle` (padded matrices)
* `teacher_counts.pickle`, `student_counts.pickle` (counts for sampling to generate masks)

### Preprocessing scripts
(not needed for training)

./preprocessing/:
1. `train_requced_tok.py` -- vocabulary trainer
2. `teacher_tokens_split.py`
3. `student2teacher.py`
4. `encode_shards.py` -- tokens to ids, outputs (teacher_ids, student_ids)
5. `prepare_lm_seqs.py` -- filter (split too long sequences, remove sequences with big unk counts, etc.)
6. `regroup_binarized.py` -- split shards to equal sizes
7. `token_counts.py` -- compute token ids in processed data
8. `init_weights.py` -- student initialization


### Training scripts
To run train (you'll probably need to change GPU count before running):
```
chmod +x run_train.sh
./run_train.sh
```
Required scripts:
1. `train.py` -- wrapper to run train
2. `distiller.py` -- trainer
3. `lm_seqs_dataset.py` -- batch generator
4. `grouped_batch_sampler.py` -- group batches by length, to reduce padding
5. `utils.py` -- auxilary utils for training
6. `setup_logger.py` -- initialize file logger for few separate scripts to be able to write logs into the same file [optional]

