# INT20H Hackathon (Data Science)
Made by **XDBoobs team** for **INT20H** 

**Main task: face clustering and aggregating images within each cluster**<br>
Train dataset: [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## General pipeline overview

### Face detection
We use ready-made solution from OpenCV for face detection and cropping. We set `minNeighbors` parameter to 45 to reduce the number of false positive instances and get more accurate face pictures. As a result, we obtain ~8.4k images for training.

### Generating vector representations from cropped images
We use [VGGFace model](https://github.com/YaleDHLab/vggface) with ResNet50 backbone to generate 2048-dimensional image embeddings. This step is done to extract informative features before clustering the images.

### Clustering
Then we perform clustering procedure using KMeans algorithm. To determine the optimal number of clusters, we carefully analyze the output of the clustering algorithm. Below you can see the resulting clusters with a general description of each:<br><br>
**Cluster 0 - Old-age man**<br>
<img src="./assets/cluster_samples/cluster_0.jpg" alt="cluster_0" width="400"/><br>
**Cluster 1 - Asian**<br>
<img src="./assets/cluster_samples/cluster_1.jpg" alt="cluster_1" width="400"/><br>
**Cluster 2 - Woman**<br>
<img src="./assets/cluster_samples/cluster_2.jpg" alt="cluster_2" width="400"/><br>
**Cluster 3 - Darker-skinned people**<br>
<img src="./assets/cluster_samples/cluster_3.jpg" alt="cluster_3" width="400"/><br>
**Cluster 4 - Young Man**<br>
<img src="./assets/cluster_samples/cluster_4.jpg" alt="cluster_4" width="400"/><br>

### Aggregating images within each cluster
To perform image aggregation we fit Conditional VAE model. CVAE is used to obtain latent distribution from which we can sample zero vector dependent on given cluster label.<br><br>
**Cluster 0**<br>
![image](https://github.com/Pelmeshek1706/inth20_hackaton/assets/94761102/eca764d6-c74a-4e71-affd-bdee75d4e528)
**Cluster 1**<br>
![image](https://github.com/Pelmeshek1706/inth20_hackaton/assets/94761102/80dcd5b2-c832-4917-8ed1-ce6660a7e827)
**Cluster 2**<br>
![image](https://github.com/Pelmeshek1706/inth20_hackaton/assets/94761102/75385894-eba4-4100-ac12-1cac00555e9e)
**Cluster 3**<br>
![image](https://github.com/Pelmeshek1706/inth20_hackaton/assets/94761102/2e602568-56ed-4190-b750-c8c84a1dfc35)
**Cluster 4**<br>
![image](https://github.com/Pelmeshek1706/inth20_hackaton/assets/94761102/0858f8f7-ca8e-494d-aad6-1f36d9828021)

## Usage guide

Python version: 3.10

1. Clone the repository:
```bash
git clone https://github.com/Pelmeshek1706/inth20_hackaton.git
```

2. Install all necessary packages:
```bash
pip install --no-cache-dir -r requirements.txt
```

### Face detection pipeline
To perform face detection and cropping pipeline run the following script:
```bash
python image_preprocessing.py --directory <path/to/directory/with/images>
```

All cropped images will be saved to the `temp` folder.

### Generating embeddings
To generate embeddings for cropped images use the following script:
```bash
python pic2vec.py [OPTIONS]
```
You may provide the following options for this script:

> | Key            | Type     | Default              | Description                                     |
> |----------------|----------|----------------------|-------------------------------------------------|
> | `image_folder` | Optional | `temp`               | Path to folder with cropped images              |
> | `csv_name`     | Optional | `image_features.csv` | Path to the CSV file to save results            |
> | `model_name`   | Optional | `resnet50`           | Name of the feature extractor/embedding model   |

This script generates CSV file with names of the images and obtained embeddings.

### Clustering
To perform KMeans clustering on generated embeddings run this script:
```bash
python clustering.py [OPTIONS]
```

> | Key               | Type     | Default               | Description                                          |
> |-------------------|----------|-----------------------|------------------------------------------------------|
> | `embeddings_path` | Required |                       | Path to the CSV files with obtained image embeddings |
> | `num_clusters`    | Required |                       | Number of clusters to form                           |
> | `output_csv`      | Optional | `cluster_results.csv` | Output file to save clustering results               |

This script generates CSV file with names of the cropped images and cluster labels.

### Image aggregating using CVAE
To fit CVAE on clustered images use the following script:
```bash
python train_cvae.py [OPTIONS]
```

> | Key           | Type     | Default | Description                                             |
> |---------------|----------|---------|---------------------------------------------------------|
> | `image_path`  | Required |         | Path to the folder with cropped images                  |
> | `labels_path` | Required |         | Path to csv file with predicted clusters for each image |
> | `num_classes` | Required |         | Number of clusters                                      |
> | `image_size`  | Optional | 224     | Resolution of train images                              |
> | `batch_size`  | Optional | 32      | Batch size for training                                 |
> | `num_epochs`  | Optional | 50      | Number of epochs to train CVAE for                      |

To run CVAE inference use this script that generates aggregated face image for the specified cluster:
```bash
python cvae_inference.py [OPTIONS]
```

> | Key           | Type     | Default                  | Description                                 |
> |---------------|----------|--------------------------|---------------------------------------------|
> | `cluster`     | Required |                          | Cluster number to generate aggregated image |
> | `weights`     | Optional | weights/cvae_epoch_21.pt | Path to the file with trained model weights |
> | `num_classes` | Optional | 5                        | Total number of clusters                    |
> | `image_size`  | Optional | 224                      | Resolution of the generated image           |

Aggregated image will be saved to the `inference_results` folder.
