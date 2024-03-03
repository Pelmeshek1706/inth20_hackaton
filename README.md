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
**Cluster 0**<br>
<img src="./assets/cluster_samples/cluster_0.jpg" alt="cluster_0" width="400"/><br>
**Cluster 1**<br>
<img src="./assets/cluster_samples/cluster_1.jpg" alt="cluster_1" width="400"/><br>
**Cluster 2**<br>
<img src="./assets/cluster_samples/cluster_2.jpg" alt="cluster_2" width="400"/><br>
**Cluster 3**<br>
<img src="./assets/cluster_samples/cluster_3.jpg" alt="cluster_3" width="400"/><br>
**Cluster 4**<br>
<img src="./assets/cluster_samples/cluster_4.jpg" alt="cluster_4" width="400"/><br>
**Cluster 5**<br>
<img src="./assets/cluster_samples/cluster_5.jpg" alt="cluster_5" width="400"/><br>
**Cluster 6**<br>
<img src="./assets/cluster_samples/cluster_6.jpg" alt="cluster_6" width="400"/><br>

### Aggregating images within each cluster
To perform image aggregation we fit Conditional VAE model. CVAE is used to obtain latent distribution from which we can sample zero vector dependent on given cluster label.<br><br>
**Cluster 0**<br>
**Cluster 1**<br>
**Cluster 2**<br>
**Cluster 3**<br>
**Cluster 4**<br>
**Cluster 5**<br>
**Cluster 6**<br>

