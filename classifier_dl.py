#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Learning Classifier evaluation.

Usage:
  deep_learning_classifier.py [--whole] [--male] [--threshold] [--leave-site-out] [--model=<model>] [--use-phenotypic] [<derivative> ...]
  deep_learning_classifier.py (-h | --help)

Options:
  -h --help              Show this screen
  --whole                Run model for the whole dataset
  --male                 Run model for male subjects
  --threshold            Run model for thresholded subjects
  --leave-site-out       Prepare data using leave-site-out method
  --model=<model>        Model to use: mlp, gan, vit [default: mlp]
  --use-phenotypic       Include phenotypic data in the model
  derivative             Derivatives to process
"""

import random
import numpy as np
import tabulate
from docopt import docopt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from utils import (load_phenotypes, format_config, hdf5_handler, load_fold, reset)


def create_mlp_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
    """Create a simple MLP model."""
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_gan_model(input_shape, latent_dim=100):
    """Create an improved GAN model."""
    # Generator
    generator = Sequential()
    generator.add(Dense(256, activation="relu", input_dim=latent_dim))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(512, activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(1024, activation="relu"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(np.prod(input_shape), activation="tanh"))
    generator.add(Reshape(input_shape))

    # Discriminator
    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=input_shape))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    # Combined model
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    return generator, discriminator, gan


def create_vit_model(input_shape, num_classes=1, learning_rate=0.001):
    """Create a Vision Transformer (ViT) model."""
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]

    inputs = layers.Input(shape=input_shape)
    patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + position_embedding

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(units=transformer_units[1], activation=tf.nn.gelu)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    for units in mlp_head_units:
        representation = layers.Dense(units, activation=tf.nn.gelu)(representation)
        representation = layers.Dropout(0.5)(representation)

    logits = layers.Dense(num_classes)(representation)
    model = Model(inputs=inputs, outputs=logits)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_model(model_type, input_shape, params):
    """Initialize the model based on type and given parameters."""
    models = {
        "mlp": create_mlp_model,
        "gan": create_gan_model,
        "vit": create_vit_model
    }
    
    model_params = {k: v for k, v in params.items() if k in models[model_type].__code__.co_varnames}
    return models[model_type](tuple(input_shape), **model_params)


def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim):
    """Train the GAN model."""
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")


def run(X_train, y_train, X_valid, y_valid, model_type="mlp", params=None, use_phenotypic=False):
    """Run a model with given parameters and return evaluation metrics."""
    
    scaler = None

    if use_phenotypic:
        scaler = StandardScaler()
        # Assuming the last four columns are the ones to scale
        X_train[:, -4:] = scaler.fit_transform(X_train[:, -4:])
        X_valid[:, -4:] = scaler.transform(X_valid[:, -4:])
    else:
        # Exclude the last 5 columns
        X_train = X_train[:, :-5]
        X_valid = X_valid[:, :-5]

    # Convert y_train and y_valid to NumPy arrays
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    input_shape = X_train.shape[1:]
    if model_type == "gan":
        generator, discriminator, gan = get_model(model_type, input_shape, params)
        train_gan(generator, discriminator, gan, X_train, epochs=params.get('epochs', 10000), batch_size=params.get('batch_size', 64), latent_dim=params.get('latent_dim', 100))
        noise = np.random.normal(0, 1, (X_valid.shape[0], params.get('latent_dim', 100)))
        pred_y = (discriminator.predict(generator.predict(noise)) > 0.5).astype("int32")
    else:
        model = get_model(model_type, input_shape, params)
        if model_type == "mlp":
            model.fit(X_train, y_train, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 32), validation_data=(X_valid, y_valid), verbose=0)
            pred_y = (model.predict(X_valid) > 0.5).astype("int32")
        elif model_type == "vit":
            # Reshape input data to 2D images (e.g., 64x64) if necessary
            side_length = int(np.sqrt(input_shape[0]))
            X_train_reshaped = X_train.reshape(-1, side_length, side_length, 1)
            X_valid_reshaped = X_valid.reshape(-1, side_length, side_length, 1)
            model.fit(X_train_reshaped, y_train, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 32), validation_data=(X_valid_reshaped, y_valid), verbose=0)
            pred_y = (model.predict(X_valid_reshaped) > 0.5).astype("int32")
    
    [[TN, FP], [FN, TP]] = confusion_matrix(y_valid, pred_y).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    fscore = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return [accuracy, precision, recall, fscore, sensitivity, specificity], scaler


def run_model(hdf5, experiment, model_type="mlp", n_iter=5, use_phenotypic=False):
    """Run model with random search on hyperparameters for each fold, only using train and validation sets."""
    param_distributions = {
        "mlp": {"learning_rate": [0.001, 0.01], "dropout_rate": [0.2, 0.5], "batch_size": [32, 64], "epochs": [10, 20]},
        "gan": {"latent_dim": [100], "epochs": [10000], "batch_size": [64]},  # Placeholder for GAN hyperparameters
        "vit": {"learning_rate": [0.001, 0.01], "epochs": [10, 20], "batch_size": [32, 64]}   # Placeholder for VIT hyperparameters
    }
    
    best_metrics = None
    best_params = None
    final_scaler = None

    # Random search over parameter samples
    for params in ParameterSampler(param_distributions[model_type], n_iter=n_iter, random_state=0):
        fold_metrics = []

        for fold in hdf5["experiments"][experiment]:
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_fold(hdf5["patients"], hdf5["experiments"][experiment], fold)

            metrics, scaler = run(X_train, y_train, X_valid, y_valid, model_type, params, use_phenotypic)
            fold_metrics.append(metrics)

        avg_metrics = np.mean(fold_metrics, axis=0).tolist()

        if best_metrics is None or avg_metrics[0] > best_metrics[0]:  # Assuming accuracy at index 0
            best_metrics = avg_metrics
            best_params = params
            final_scaler = scaler  # Save the scaler used for the best model

    return best_metrics, best_params, final_scaler


# Main function
if __name__ == "__main__":
    reset()
    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)
    hdf5 = hdf5_handler(bytes("./data/abide.hdf5", encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [d for d in arguments["<derivative>"] if d in valid_derivatives]
    experiments = []

    for derivative in derivatives:
        config = {"derivative": derivative}
        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)]
        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]
        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]
        
        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                site_config = {"site": site}
                experiments += [format_config("{derivative}_leavesiteout-{site}", config, site_config)]

    experiments = sorted(experiments)
    experiment_results = []
    best_params_results = {}

    use_phenotypic = arguments["--use-phenotypic"]

    for experiment in experiments:
        model_type = arguments["--model"]
        metrics, best_params, scaler = run_model(hdf5, experiment, model_type, use_phenotypic=use_phenotypic)
        experiment_results.append([experiment] + metrics)
        best_params_results[experiment] = best_params  # Store best params for each experiment

    # Display all results in a table
    print(tabulate.tabulate(experiment_results, headers=["exp", "acc", "prec", "recall", "fscore", "sens", "spec"]))

    # Display the best parameters for each experiment
    print("\nBest parameters for each experiment:")
    for experiment, best_params in best_params_results.items():
        print(f"Experiment: {experiment}, Best Parameters: {best_params}")