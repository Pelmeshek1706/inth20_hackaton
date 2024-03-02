import cv2
import os
from tqdm import tqdm
from argparse import ArgumentParser


def _crop_image_by_bbox(image, bbox):
    """
    Crops an image based on a bounding box (x, y, width, height).

      Args:
          image: The image to be cropped (NumPy array).
          bbox: A tuple containing bounding box coordinates (x, y, width, height).

      Returns:
          A cropped image (NumPy array) or None if the bounding box is invalid.
    """

    # Extract bounding box coordinates
    x, y, w, h = bbox

    # Check for valid bounding box (entirely within image)
    if (x >= 0) and (y >= 0) and (x + w <= image.shape[1]) and (y + h <= image.shape[0]):
        return image[y:y + h, x:x + w]  # Slice the image using bounding box coordinates
    else:
        print("Invalid bounding box. Image not cropped.")
        return None  # Return None if bounding box is outside image


def _detect_face(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=45)

    if len(faces_rect) > 0:
        for (x, y, w, h) in faces_rect:
            # Crop the face using the bounding box
            cropped_face = _crop_image_by_bbox(img, (x, y, w, h))
        return cropped_face
    else:
        return None


def _save_picture(image, filename):
    """
    Save the image to the temp directory
    :param image:
    :param filename:
    :return:
    """
    if not os.path.exists("kek"):
        os.makedirs("kek")
    cv2.imwrite(os.path.join('kek', filename), image)


def _check_image_size(image_path):
    """
    Check image size

    Args:
        image_path (str): Path to the image

    Returns:
        bool: Whether the image size is greater than 100 px
    """
    img = cv2.imread(image_path)
    if img is not None:
        height, width, _ = img.shape
        return width >= 100 and height >= 100
    return False


def process_directory(directory):
    """
    Process images from the directory and save them to 'temp' folder

    Args:
        directory (str): Path to the directory containing images
    """
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' doesn't exist")
        return

    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                if _check_image_size(file_path):
                    crop_face = _detect_face(file_path)
                    if crop_face is None:
                        continue
                    else:
                        _save_picture(crop_face, file)
                        
                        
def main():
    parser = ArgumentParser()
    parser.add_argument("--directory", type=str, help="Path to the directory with images")
    args = parser.parse_args()
    directory = args.directory

    process_directory(directory)


if __name__ == "__main__":
    main()