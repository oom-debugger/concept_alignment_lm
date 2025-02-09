

import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-shared_vocab_size", "--shared_vocab_size", dest = "shared_vocab_size", default = 15, help="shared vocab size.", type=int)


def main():
  args = parser.parse_args()
  for k in [3, 5, 10, 50, 100]:
    alignment_scores = []
    for _ in range(1000):
        x1 = np.random.choice(args.shared_vocab_size, k)
        x2 = np.random.choice(args.shared_vocab_size, k)
        alignment_scores.append(len(set(x1.tolist()).intersection(x2.tolist())) / k)
    print (f'Baseline alignment score for k={k} is {np.mean(alignment_scores)} for shared vocab size of {args.shared_vocab_size}.')


if __name__ == "__main__":
    main()
