import csv
import numpy as np
from sklearn.cluster import KMeans
from argparse import ArgumentParser


def _load_embeddings(embeddings_path):
    embeddings = []
    filenames = []
    with open(embeddings_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header row
        for row in reader:
            filenames.append(row[0])
            embeddings.append(
                [float(x) for x in row[1].split(',')]  # Convert string features to floats
            )

    embeddings = np.array(embeddings)
    return filenames, embeddings


def _kmeans_clustering(num_clusters, embeddings):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_.tolist()
    return cluster_labels


def _generate_clustering_results(filenames, cluster_labels, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'class'])

        for image_name, cluster_id in zip(filenames, cluster_labels):
            writer.writerow([image_name, cluster_id])


def main():
    parser = ArgumentParser()
    parser.add_argument('--embeddings_path', type=str)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--output_csv', type=str, required=False, default='cluster_results.csv')
    args = parser.parse_args()

    filenames, embeddings = _load_embeddings(embeddings_path=args.embeddings_path)
    cluster_labels = _kmeans_clustering(num_clusters=args.num_clusters, embeddings=embeddings)
    _generate_clustering_results(filenames=filenames,
                                 cluster_labels=cluster_labels,
                                 output_csv=args.output_csv)


if __name__ == '__main__':
    main()
