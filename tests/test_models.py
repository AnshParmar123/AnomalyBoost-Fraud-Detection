import pytest
import tensorflow as tf
from src.models import build_dense_autoencoder, build_vae

def test_dense_autoencoder_shape():
    input_dim = 30
    encoding_dim = 8
    model = build_dense_autoencoder(input_dim, encoding_dim)
    
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, input_dim)
    
    # Bottleneck layer checks
    bottleneck_layer = model.get_layer('bottleneck')
    assert bottleneck_layer.output.shape == (None, encoding_dim)

def test_vae_shape():
    input_dim = 30
    latent_dim = 8
    model = build_vae(input_dim, latent_dim)
    
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, input_dim)
