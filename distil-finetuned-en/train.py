from argparse import ArgumentParser
from distiller import Distiller
from trainer import StudentTrainer
import json
import ruamel.yaml as yaml
import os
import wandb
import shutil


parser = ArgumentParser()
parser.add_argument('--glue_dataset', choices=['cola', 'sst2', 'mrpc', 
                                               'qqp', 'stsb', 'mnli', 
                                               'qnli', 'rte', 'wnli'])

tokenizer_group = parser.add_argument_group('tokenizer group')
tokenizer_group.add_argument('--tokenizer_name')
tokenizer_group.add_argument('--tokenizer_params', help="""yaml file with tokenizer params""")

training_options = parser.add_argument_group('training_options')
training_options.add_argument('--batch_size', type=int, default=4)
training_options.add_argument('--n_epochs', type=int, default=1)
training_options.add_argument('--lr', type=float, default=1e-3)

"""
training_options.add_argument('--scheduler', type=str)
training_options.add_argument('--scheduler_params')
"""

training_options.add_argument('--val_every_n_batches', type=int, default=1)
training_options.add_argument('--val_after_epoch', action='store_true')

training_options.add_argument('--gpu_id', type=int, default=0)
training_options.add_argument('--alpha_ce', type=float, default=1.0)
training_options.add_argument('--seed', type=int, default=42)

training_options.add_argument('--log_name', type=str)
training_options.add_argument('--dumps_dir', type=str)

training_options.add_argument('--valid_prop', type=float, default=0.1)
training_options.add_argument('--valid_patience', type=int, default=1)
training_options.add_argument('--lr_drop_patience', type=int, default=1)
training_options.add_argument('--lr_drop_div', type=float, default=2)
training_options.add_argument('--min_lr', type=float, default=1e-10)

wandb_options = parser.add_argument_group('wandb options')
wandb_options.add_argument('--wandb_config', help='.yaml file with project name, run name, mode, etc.')
wandb_options.add_argument('--run_id')

subparsers = parser.add_subparsers(help="""Running mode""", dest='mode')


train_student = subparsers.add_parser('train_student')
train_student.add_argument('--student_name')
train_student.add_argument('--from_pretrained', action='store_true')

distil = subparsers.add_parser('distil_teacher')
distil.add_argument('--teacher_name', type=str)
distil.add_argument('--teacher_weights', type=str)
distil.add_argument('--student_name', type=str)

distil.add_argument('--project_to', choices=['teacher', 'student', 'intermediate', None], default=None)
distil.add_argument('--intermediate_dim', type=int)
distil.add_argument('--projection_strategy', choices=['last', 'skip', 'average', 'average_by_layers', 'select_by_ids'], default='average_by_layers')
distil.add_argument('--t_s_layers_ids', type=json.loads)

kl_options = distil.add_argument_group('kl options')
kl_options.add_argument('--alpha_kl', type=float, default=0.0)
kl_options.add_argument('--temperature', type=float, default=1.0)
kl_options.add_argument('--train_temperature', action='store_true')

contrastive_options = distil.add_argument_group('contrastive options')
contrastive_options.add_argument('--alpha_contrastive', type=float, default=0.0)
contrastive_options.add_argument('--from_one_sample', action='store_true')
contrastive_options.add_argument('--n_negative_samples', type=int, default=-1)
contrastive_options.add_argument('--teacher_student_prop', nargs='?', type=float, default=0.5)
contrastive_options.add_argument('--negative_sampling_strategy', choices=['teacher', 'student', 'teacher_and_student', None], default=None)
contrastive_options.add_argument('--add_neg_size_constant', action='store_true')

mse_options = parser.add_argument_group('mse options')
mse_options.add_argument('--alpha_mse', type=float, default=0.0)


distil_subparsers = distil.add_subparsers(dest='hidden_distil_type')

hyperbolic = distil_subparsers.add_parser('hyperbolic')
hyperbolic.add_argument('--c', type=float, default=1.0)
hyperbolic.add_argument('--init_c', 
                        choices=['precompute_from_teacher', 'precompute_from_student', None], 
                        default=None, 
                        help="precompute* -- precompute curvature from teacher/student outputs via Gromov product")
hyperbolic.add_argument('--n_samples_to_precompute_c', type=int, default=100)

hyperbolic.add_argument('--reduce_to_n_components', type=int, default=-1, 
                        help="""reduce output feature space (vocab_size) to n_components; 
                        works with `init_c=precompute*` or `adjust_c=recompute_after_epoch`""")
hyperbolic.add_argument('--n_tries', type=int, default=1, 
                        help="""number of sampling tries to precompute delta and diam, 
                        works with `init_c=precompute*` or `adjust_c=recompute_after_epoch`""")
hyperbolic.add_argument('--train_c', choices=['train_exp_map_teacher', 
                                               'train_exp_map_student', 
                                               'recompute_after_epoch', 
                                                None], default=None)
hyperbolic.add_argument('--riemannian', action='store_false')
hyperbolic.add_argument('--train_x', action='store_true')

hyplinear = hyperbolic.add_argument_group('options for hyperbolic linear layers')
hyplinear.add_argument('--use_hyperbolic_projections', action='store_true')
hyplinear.add_argument('--use_bias', action='store_false')


args, _ = parser.parse_known_args()

with open(args.wandb_config) as f:
    args.wandb_config = yaml.load(f)
os.environ['WANDB_API_KEY'] = args.wandb_config['WANDB_API_KEY']
os.environ['WANDB_DIR'] = args.wandb_config['WANDB_DIR']
if os.path.exists(args.dumps_dir):
    shutil.rmtree(args.dumps_dir)
os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
os.makedirs(args.dumps_dir, exist_ok=True)
os.environ['WANDB_ENTITY'] = args.wandb_config['WANDB_ENTITY']
os.environ['WANDB_MODE'] = args.wandb_config['WANDB_MODE']

config = vars(args)

args.run = wandb.init(reinit=args.wandb_config['reinit'], 
                      id=args.run_id, 
                      project=args.wandb_config['project'], 
                      config=config)
"""
with open(args.scheduler_params) as f:
    args.scheduler_params = yaml.load(f)
"""

with open(args.tokenizer_params) as f:
    args.tokenizer_params = yaml.load(f)

if 'train' in args.mode:
    trainer = StudentTrainer(args)
    trainer.train()
elif 'distil' in args.mode:
    distiller = Distiller(args)
    distiller.train()



