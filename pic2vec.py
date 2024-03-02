import os
import csv
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from tqdm import tqdm

from argparse import ArgumentParser

def main():

    parser = ArgumentParser(description="generate vector embeddings")

    parser.add_argument('--image_folder',
                            type=str,
                            default='temp')

    parser.add_argument("--csv_name", 
                            type=str,
                            default="image_features.csv")

    parser.add_argument('--model_name',
                            type=str,
                            default='resnet50')

    args = parser.parse_args()

    # Load VGGFace model
    vgg_features = VGGFace(model=args.model_name, include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Open CSV file for writing
    with open(args.csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(['filename', 'features'])

        # Loop through all images in the folder
        image_folder = args.image_folder
        for filename in tqdm(os.listdir(image_folder)):

            # Get the full image path
            image_path = os.path.join(image_folder, filename)

            # Load image and preprocess
            try:
                img = image.load_img(image_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                # x = utils.preprocess_input(x, version=2)  # Optional preprocessing (uncomment if needed)
            except Exception as e:
                print(f"Error processing image: {image_path}. Skipping.")
                continue

            # Extract features
            features = vgg_features.predict(x)[0]

            # Convert features to comma-separated string
            features_string = ','.join(map(str, features.tolist()))  # Convert each feature to string and join

            # Write filename and features string to CSV
            writer.writerow([filename, features_string])

    print(f"Image features extracted and saved to: {csv_file}")

if __name__ == "__main__":
    main()