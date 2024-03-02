# inth20_hackaton
# Made by **XDBoobs team** for **int20h** 

## Main task : generate mean person for each cluster from [dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)


<!-- 
### That we get at the start point: 
images with faces
// 5 pics from dataset
### That we generate -->

### At first - decompose the task
- First step: [Face detection](link for demo.ipynb)
    We using ready-made solution from [OpenCV](https://www.datacamp.com/tutorial/face-detection-python-opencv) for Face-recognition and after processing our data we recieve ~8.4k cropped images.
    Folder -> Folder
- Second step: [Vector Embeddings](link pic2vec.py)
    Taking our directory with cropped images and vectorize it to 2048 dimensional using [VGGFace](https://github.com/YaleDHLab/vggface) 
    Folder -> SCV file
- Third Step: [Clustering](link for kmeans.py)
    We looked at the informativeness of the clusters and by long logical reasoning we came up with 7 clusters 
    (pics from t for this clusters)
- Generating mean person from each cluster : [cvae.py](link for cvae)
    ///здесь что-то про СВАЕ я не знаю 
