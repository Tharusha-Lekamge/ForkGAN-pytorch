import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import glob
import random


def calculate_fid_score(real_images_path, fake_images_path, num_images=2000):
    # Load pre-trained InceptionV3 model
    inception_model = InceptionV3(
        include_top=False, pooling="avg", input_shape=(299, 299, 3)
    )

    # Feature extraction model
    feature_extractor = Model(
        inputs=inception_model.input, outputs=inception_model.layers[-1].output
    )

    def preprocess_images(image_paths):
        images = []
        for img_path in image_paths:
            img = image.load_img(img_path, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images.append(img)
        return np.vstack(images)

    def calculate_activation_statistics(images):
        features = feature_extractor.predict(images)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
        # Calculate the sum of squared differences between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # Calculate the square root of the product between covariances
        covmean = sqrtm(sigma1.dot(sigma2))
        # Check for imaginary numbers and discard them
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Calculate the Frechet distance
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # Load and preprocess random subset of images
    real_image_paths = random.sample(glob.glob(real_images_path), num_images)
    fake_image_paths = random.sample(glob.glob(fake_images_path), num_images)
    real_images = preprocess_images(real_image_paths)
    fake_images = preprocess_images(fake_image_paths)
    print("Real Images Shape:", real_images.shape)
    print("Fake Images Shape:", fake_images.shape)

    # Calculate statistics for real and fake images
    mu1, sigma1 = calculate_activation_statistics(real_images)
    mu2, sigma2 = calculate_activation_statistics(fake_images)
    # print("Mean and Covariance of real images:", mu1, sigma1)
    # print("Mean and Covariance of fake images:", mu2, sigma2)

    # Calculate FID score
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score


# Example usage:
real_images_path = "datasets/merged_ds/testB/*.jpg"
fake_images_path = "results/bdd100k-complete/fid/fake/*.png"
fid_score = calculate_fid_score(real_images_path, fake_images_path)
print("FID Score:", fid_score)