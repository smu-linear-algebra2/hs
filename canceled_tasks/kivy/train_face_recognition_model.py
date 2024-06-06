import math
from sklearn import neighbors
import os
import pickle
from PIL import Image
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print(f"Image {img_path} not suitable for training: {len(face_bounding_boxes)} faces found.")
                continue
            else:
                face_enc = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                X.append(face_enc)
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print(f"Chose n_neighbors automatically: {n_neighbors}")

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a KNN classifier for face recognition")
    parser.add_argument("train_dir", type=str, help="Directory with sub-directories of images for each person")
    parser.add_argument("model_save_path", type=str, help="Path to save the trained KNN model")
    parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for KNN", default=None)
    parser.add_argument("--knn_algo", type=str, help="KNN algorithm to use", default='ball_tree')
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    print("Training KNN classifier...")
    classifier = train(args.train_dir, model_save_path=args.model_save_path, n_neighbors=args.n_neighbors, knn_algo=args.knn_algo, verbose=args.verbose)
    print("Training complete. Model saved to", args.model_save_path)
