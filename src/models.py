import logging
import keras
from keras import layers, models, ops

logger = logging.getLogger(__name__)


class Sampling(layers.Layer):
    """Reparameterization layer that also registers KL divergence loss."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = ops.shape(z_mean)[0]
        latent_dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
        z = z_mean + ops.exp(0.5 * z_log_var) * epsilon

        kl_loss = -0.5 * ops.mean(1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        self.add_loss(kl_loss)
        return z

def build_dense_autoencoder(input_dim: int, encoding_dim: int) -> models.Model:
    """Builds a symmetric dense feedforward autoencoder."""
    logger.info(f"Building Dense Autoencoder: input_dim={input_dim}, encoding_dim={encoding_dim}")
    
    inp = layers.Input(shape=(input_dim,), name="ae_input")
    x = layers.Dense(max(256, input_dim), activation='relu', name="encoder_dense_1")(inp)
    x = layers.Dense(128, activation='relu', name="encoder_dense_2")(x)
    x = layers.Dense(64, activation='relu', name="encoder_dense_3")(x)
    
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)
    
    x = layers.Dense(64, activation='relu', name="decoder_dense_1")(encoded)
    x = layers.Dense(128, activation='relu', name="decoder_dense_2")(x)
    x = layers.Dense(max(256, input_dim), activation='relu', name="decoder_dense_3")(x)
    out = layers.Dense(input_dim, activation='linear', name="ae_output")(x)
    
    model = models.Model(inputs=inp, outputs=out, name="Dense_Autoencoder")
    model.compile(optimizer='adam', loss='mse')
    return model

def build_vae(input_dim: int, latent_dim: int = 16) -> models.Model:
    """Builds a Variational Autoencoder compatible with Keras 3."""
    logger.info(f"Building Variational Autoencoder (VAE): input_dim={input_dim}, latent_dim={latent_dim}")
    
    inputs = layers.Input(shape=(input_dim,), name="vae_input")
    x = layers.Dense(128, activation='relu', name="vae_enc_dense_1")(inputs)
    x = layers.Dense(64, activation='relu', name="vae_enc_dense_2")(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling(name='z_sampling')([z_mean, z_log_var])

    decoder_h1 = layers.Dense(64, activation='relu', name="vae_dec_dense_1")
    decoder_h2 = layers.Dense(128, activation='relu', name="vae_dec_dense_2")
    decoder_out = layers.Dense(input_dim, activation='linear', name="vae_output")

    h_decoded = decoder_h1(z)
    h_decoded = decoder_h2(h_decoded)
    x_decoded_mean = decoder_out(h_decoded)

    vae = models.Model(inputs, x_decoded_mean, name="Variational_Autoencoder")
    vae.compile(optimizer='adam', loss='mse')
    return vae
