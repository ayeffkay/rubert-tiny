import argparse
import logging
import pickle
from collections import Counter
import glob


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)"
    )
    parser.add_argument(
        "--binarized_data_folder", type=str, default="data/dump.bert-base-uncased.pickle", help="The binarized dataset."
    )
    parser.add_argument(
        "--data_column", type=int
    )
    parser.add_argument(
        "--token_counts_dump", type=str, default="data/token_counts.bert-base-uncased.pickle", help="The dump file."
    )
    parser.add_argument("--vocab_size", default=30522, type=int)
    
    args = parser.parse_args()

    logger.info(f"Loading data from {args.binarized_data_folder}")
    data = []
    for file in glob.glob(f'{args.binarized_data_folder}/*'):
        with open(file, 'rb') as f:
            shard = pickle.load(f)
            shard = [token for seq in shard for token in seq[args.data_column]]
            data.extend(shard)

    logger.info("Counting occurences for MLM.")
    counter = Counter(data)
    
    counts = [0] * args.vocab_size
    for k, v in counter.items():
        counts[k] = v

    logger.info(f"Dump to {args.token_counts_dump}")
    with open(args.token_counts_dump, "wb") as handle:
        pickle.dump(counts, handle)

