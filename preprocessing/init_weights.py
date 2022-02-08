
from argparse import ArgumentParser
import pickle
from transformers import (DistilBertConfig, BertConfig, AutoModelForSequenceClassification,
                          BertForMaskedLM, DistilBertForMaskedLM)
from collections import defaultdict
import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mapping', nargs='?')
    parser.add_argument('--teacher_name')
    parser.add_argument('--teacher_weights', nargs='?')
    parser.add_argument('--student_name')
    parser.add_argument('--dump_checkpoint')
    parser.add_argument('--vocab_transform', action='store_true')
    parser.add_argument('--mode', choices=['masked_lm', 'finetuning'])
    parser.add_argument('--num_labels', type=int, default=2, nargs='?')
    
    args, _ = parser.parse_known_args()

    if args.mode == 'masked_lm':
        teacher_model = BertForMaskedLM.from_pretrained(args.teacher_name)
        student_config = DistilBertConfig.from_pretrained(args.student_name)
        student_model = DistilBertForMaskedLM(student_config)
    if args.mode == 'finetuning':
        student_config = DistilBertConfig.from_pretrained(args.student_name, num_labels=args.num_labels)
        teacher_config = BertConfig.from_pretrained(args.teacher_name, num_labels=args.num_labels)

        teacher_model = AutoModelForSequenceClassification.from_config(teacher_config)
        teacher_model.load_state_dict(torch.load(args.teacher_weights))
        student_model = AutoModelForSequenceClassification.from_config(student_config)

    teacher_sd = teacher_model.state_dict()
    student_sd = student_model.state_dict()

    student_embs_name = 'distilbert.embeddings.word_embeddings.weight'
    teacher_embs_name = 'bert.embeddings.word_embeddings.weight'

    student_pos_ids_name = 'distilbert.embeddings.position_embeddings.weight'
    teacher_pos_ids_name = 'bert.embeddings.position_embeddings.weight'
    
    if args.mapping is not None:
        with open(args.mapping, 'rb') as f:
            mapping_dict = pickle.load(f)

        student2teacher = defaultdict(list)
        for teacher_id, student_ids in mapping_dict.items():
            for student_id in student_ids:
                if teacher_id not in student2teacher[student_id]:
                    student2teacher[student_id].append(teacher_id)

        for i in range(student_config.vocab_size):
            teacher_ids = student2teacher[i]
            if len(teacher_ids):
                teacher_mean_embs = torch.stack([teacher_sd[teacher_embs_name][j] for j in teacher_ids]).mean(dim=0)
                student_sd[student_embs_name][i] = teacher_mean_embs[:student_config.dim]
            
        student_sd[student_pos_ids_name] = teacher_sd[teacher_pos_ids_name][:, :student_config.dim]
        for i in range(student_config.vocab_size):
            if len(student2teacher[i]):
                teacher_w = torch.stack([teacher_sd['cls.predictions.decoder.weight'][j] for j in student2teacher[i]]).mean(dim=0)
                student_sd['vocab_projector.weight'][i] = teacher_w[:student_config.dim]
        for i in range(student_config.vocab_size):
            if len(student2teacher[i]):
                teacher_w = torch.tensor([teacher_sd['cls.predictions.decoder.bias'][j] for j in student2teacher[i]]).mean(dim=0)
                student_sd['vocab_projector.bias'][i] = teacher_w
    else:
        student_sd[student_embs_name] = teacher_sd[teacher_embs_name][:, :student_config.dim]
        student_sd[student_pos_ids_name] = teacher_sd[teacher_pos_ids_name][:, :student_config.dim]
        if args.mode != 'finetuning':
            student_sd["vocab_projector.weight"] = teacher_sd["cls.predictions.decoder.weight"][:, :student_config.dim]
            student_sd["vocab_projector.bias"] = teacher_sd["cls.predictions.bias"]
    
    for w in ['weight', 'bias']:
        student_layer_norm_name = f'distilbert.embeddings.LayerNorm.{w}'
        teacher_layer_norm_name = f'bert.embeddings.LayerNorm.{w}'
        student_sd[student_layer_norm_name] = teacher_sd[teacher_layer_norm_name][:student_config.dim]
    
    teacher_q_lin = []
    teacher_k_lin = []
    teacher_v_lin = []
    teacher_out_lin = []
    teacher_out_layer_norm = []
    teacher_intermediate = []
    teacher_out_dense = []
    teacher_out_layer_norm_ = []
    
    for w in ['weight', 'bias']:
        for i in range(12):
            teacher_q_lin.append(
                teacher_sd[f'bert.encoder.layer.{i}.attention.self.query.{w}']) 
            teacher_k_lin.append(
                teacher_sd[f'bert.encoder.layer.{i}.attention.self.key.{w}'])
            teacher_v_lin.append(
                teacher_sd[f'bert.encoder.layer.{i}.attention.self.value.{w}'])
            teacher_out_lin.append(
                teacher_sd[f'bert.encoder.layer.{i}.attention.output.dense.{w}'])
            teacher_out_layer_norm.append(
                teacher_sd[f'bert.encoder.layer.{i}.attention.output.LayerNorm.{w}']
            )
            teacher_intermediate.append(
                teacher_sd[f'bert.encoder.layer.{i}.intermediate.dense.{w}']
            )
            teacher_out_dense.append(
                teacher_sd[f'bert.encoder.layer.{i}.output.dense.{w}']
            )
            teacher_out_layer_norm_.append(
                teacher_sd[f'bert.encoder.layer.{i}.output.LayerNorm.{w}']
            )
            
            if (i + 1) % 4 == 0:
                teacher_q_lin = torch.stack(teacher_q_lin).mean(dim=0)
                teacher_k_lin = torch.stack(teacher_k_lin).mean(dim=0)
                teacher_v_lin = torch.stack(teacher_v_lin).mean(dim=0)
                teacher_out_lin = torch.stack(teacher_out_lin).mean(dim=0)
                teacher_out_layer_norm = torch.stack(teacher_out_layer_norm).mean(dim=0)[:student_config.dim]
                teacher_intermediate = torch.stack(teacher_intermediate).mean(dim=0)
                teacher_out_dense = torch.stack(teacher_out_dense).mean(dim=0)
                teacher_out_layer_norm_ = torch.stack(teacher_out_layer_norm_).mean(dim=0)[:student_config.dim]

                if w == 'weight':
                    teacher_q_lin = teacher_q_lin[:student_config.dim, :student_config.dim]
                    teacher_k_lin = teacher_k_lin[:student_config.dim, :student_config.dim]
                    teacher_v_lin = teacher_v_lin[:student_config.dim, :student_config.dim]
                    teacher_out_lin = teacher_out_lin[:student_config.dim, :student_config.dim]
                    teacher_intermediate = teacher_intermediate[:student_config.hidden_dim, :student_config.dim]
                    teacher_out_dense = teacher_out_dense[:student_config.dim, :student_config.hidden_dim]
                else:
                    teacher_q_lin = teacher_q_lin[:student_config.dim]
                    teacher_k_lin = teacher_k_lin[:student_config.dim]
                    teacher_v_lin = teacher_v_lin[:student_config.dim]
                    teacher_out_lin = teacher_out_lin[:student_config.dim]
                    teacher_intermediate = teacher_intermediate[:student_config.hidden_dim]
                    teacher_out_dense = teacher_out_dense[:student_config.dim]

                student_sd[f'distilbert.transformer.layer.{i // 4}.attention.q_lin.{w}'] = teacher_q_lin
                student_sd[f'distilbert.transformer.layer.{i // 4}.attention.k_lin.{w}'] = teacher_k_lin
                student_sd[f'distilbert.transformer.layer.{i // 4}.attention.v_lin.{w}'] = teacher_v_lin
                student_sd[f'distilbert.transformer.layer.{i // 4}.attention.out_lin.{w}'] = teacher_out_lin
                student_sd[f'distilbert.transformer.layer.{i // 4}.sa_layer_norm.{w}'] = teacher_out_layer_norm
                student_sd[f'distilbert.transformer.layer.{i // 4}.ffn.lin1.{w}'] = teacher_intermediate
                student_sd[f'distilbert.transformer.layer.{i // 4}.ffn.lin2.{w}'] = teacher_out_dense
                student_sd[f'distilbert.transformer.layer.{i // 4}.output_layer_norm.{w}'] = teacher_out_layer_norm_

                teacher_q_lin = []
                teacher_k_lin = []
                teacher_v_lin = []
                teacher_out_lin = []
                teacher_out_layer_norm = []
                teacher_intermediate = []
                teacher_out_dense = []
                teacher_out_layer_norm_ = []
        
    if args.vocab_transform:
        student_sd['vocab_transform.weight'] = teacher_sd['cls.predictions.transform.dense.weight'][:student_config.dim, :student_config.dim]
        student_sd['vocab_transform.bias'] = teacher_sd['cls.predictions.transform.dense.bias'][:student_config.dim]
        student_sd['vocab_layer_norm.weight'] = teacher_sd['cls.predictions.transform.LayerNorm.weight'][:student_config.dim]
        student_sd['vocab_layer_norm.bias'] = teacher_sd['cls.predictions.transform.LayerNorm.bias'][:student_config.dim]
    
    torch.save(student_sd, args.dump_checkpoint)    
    student_model.load_state_dict(torch.load(args.dump_checkpoint))