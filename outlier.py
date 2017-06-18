from sklearn.ensemble import IsolationForest

import numpy as np
import sys


def find_outliers(input_x, ratio):
    assert len(input_x.shape) == 2
    forest = IsolationForest(max_samples=min(100, len(input_x)),
                             contamination=ratio)
    forest.fit(input_x)
    return forest.predict(input_x)


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("{} <input_file> <output_file>\n".format(argv[0]))
        sys.exit(1)

    # read input embeddings
    with open(argv[1], 'rt') as in_fn, open(argv[2], 'wt') as out_fn:
        embeddings = []
        for in_line in in_fn:
            if in_line.startswith("<id"):
                if len(embeddings) > 0:
                    inliers = find_outliers(np.asarray(embeddings), 0.2)
                    for inlier in inliers:
                        out_fn.write("{}\n".format(inlier))
                    embeddings = []
                out_fn.write(in_line)
            else:
                embedding = np.array(in_line.strip().split(" "), np.float32)
                embeddings.append(embedding)

if __name__ == '__main__':
    main(sys.argv)
