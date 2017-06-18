import numpy as np
import outlier


def embedding_avg(file_path, rm_outliers=True):
    embeddings = []
    ids = []
    id_ = "uninitialized id"
    avg_embeddings = []
    with open(file_path) as fn:
        for line in fn:
            if line.startswith("<id"):
                if len(embeddings) > 0:
                    if rm_outliers:
                        outlier_labels = outlier.find_outliers(
                            np.asarray(embeddings), 0.2)
                        embeddings_ = []
                        for i in xrange(len(outlier_labels)):
                            if outlier_labels[i] == 1:
                                embeddings_.append(embeddings[i])
                        embeddings = embeddings_
                    if len(embeddings) > 0:
                        avg_embedding = np.average(np.asarray(embeddings), axis=0)
                        avg_embeddings.append(avg_embedding)
                        ids.append(id_)
                    embeddings = []
                id_ = line.strip()
            else:
                embedding = np.array(line.strip().split(" "), np.float32)
                embeddings.append(embedding)
    return ids, avg_embeddings

