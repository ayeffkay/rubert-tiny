This is official implementation of the paper [**`Knowledge Distillation of Russian Language Models with Reduction of Vocabulary`**](https://arxiv.org/abs/2205.02340).

### Citation
If you found this code or results from the paper useful, we are kindly ask your to cite this paper:
```
@misc{https://doi.org/10.48550/arxiv.2205.02340,
  doi = {10.48550/ARXIV.2205.02340},
  
  url = {https://arxiv.org/abs/2205.02340},
  
  author = {Kolesnikova, Alina and Kuratov, Yuri and Konovalov, Vasily and Burtsev, Mikhail},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Knowledge Distillation of Russian Language Models with Reduction of Vocabulary},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
### Credits
This code is based on [DistilBERT](https://arxiv.org/abs/1910.01108) official implementation https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation.

### Dependencies
```
pip install -r requirements.txt
```
### Data
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
9. `find_matching_ids.py` -- add mask with matching teacher and student matching ids 
10. `matched_vocab.py` -- outputs dict of the form `{matched_vocab_token: [teacher_id, student_id]}`


### Training scripts
To run train (you'll probably need to change GPU count before running):
```
chmod +x {script_name}.sh
./{script_name}.sh
```
Required scripts:
1. `train.py` -- wrapper to run train
2. `distiller.py` -- trainer
3. `lm_seqs_dataset.py` -- batch generator
4. `custom_step.py` -- functions one train/valid step with different losses
5. `my_index.py` -- backward optimization
6. `grouped_batch_sampler.py` -- group batches by length, to reduce padding
7. `utils.py` -- auxilary utils for training
8. `setup_logger.py` -- initialize file logger for few separate scripts to be able to write logs into the same file [optional]

## Hyperbolic scripts
1. `delta.py` -- functions to precompute curvature
2. `hyptorch/` -- hyperbolic layers and related stuff

## GLUE
`distil-finetuned-en` folder contains scripts for fine-tuning teachers on GLUE and distillation


