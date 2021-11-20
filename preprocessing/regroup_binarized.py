from argparse import ArgumentParser
import pickle
import glob
from pathlib import Path
import shutil


def main():
    parser = ArgumentParser()
    parser.add_argument('--binarized_folder')
    parser.add_argument('--n_shards', type=int)
    args, _ = parser.parse_known_args()
    
    data = []
    old_files = sorted(glob.glob(f'{args.binarized_folder}/*'))
    for file in old_files:
        with open(file, 'rb') as f:
            shard = pickle.load(f)
            data.extend(shard)
    samples_per_shard = len(data) // args.n_shards
    for i in range(args.n_shards):
        shutil.rmtree(old_files[i])
        with open(Path(args.binarized_folder)/f'shard_{i}.pickle', 'wb') as f:
            pickle.dump(data[i * samples_per_shard: (i + 1) * samples_per_shard], f)
    

if __name__ == '__main__':
    main()
