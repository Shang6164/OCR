import os
import json
from sklearn.model_selection import train_test_split

def load_data(image_dir, annotation_dir):
    """
    Load images and annotations from the specified directories.
    
    Args:
        image_dir (str): Path to the directory containing images.
        annotation_dir (str): Path to the directory containing annotations.

    Returns:
        list: A list of tuples (image_path, annotation_path).
    """
    data = []
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.json'):
            annotation_path = os.path.join(annotation_dir, annotation_file)
            image_file = annotation_file.replace('.json', '.jpg')
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(image_path):
                data.append((image_path, annotation_path))
    return data

def split_data(data, test_size=0.2):
    """
    Split data into training and testing sets.

    Args:
        data (list): List of data tuples (image_path, annotation_path).
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing data splits.
    """
    return train_test_split(data, test_size=test_size, random_state=42)

if __name__ == "__main__":
    IMAGE_DIR = "../card_cropping_data/images"
    ANNOTATION_DIR = "../card_cropping_data/annotations"

    # Load and split data
    data = load_data(IMAGE_DIR, ANNOTATION_DIR)
    train_data, test_data = split_data(data)

    print(f"Total data: {len(data)}")
    print(f"Training data: {len(train_data)}")
    print(f"Testing data: {len(test_data)}")