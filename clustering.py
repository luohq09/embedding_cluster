import sys
from sklearn.cluster import KMeans
import data_helper


def kmeans_cluster(x):
    num_cluster = len(x) / 5
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(x)
    return kmeans.labels_


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("{} <input_file> <output_file>\n".format(argv[0]))
        sys.exit(1)

    ids, avg_embeddings = data_helper.embedding_avg(argv[1])
    cluster_labels = kmeans_cluster(avg_embeddings)
    clusters = {}
    for i in xrange(len(ids)):
        cluster_label = cluster_labels[i]
        id_ = ids[i]
        if cluster_label in clusters:
            clusters[cluster_label].append(id_)
        else:
            clusters[cluster_label] = [id_]

    with open(argv[2], 'wt') as out_fn:
        for k, v in clusters.iteritems():
            out_fn.write("cluster_{}\n".format(k))
            for id_ in v:
                out_fn.write("{}\n".format(id_))

if __name__ == '__main__':
    main(sys.argv)