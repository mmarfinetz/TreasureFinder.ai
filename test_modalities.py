import numpy as np

from treasure_hunter_module import fetch_satellite_image, NUM_CHANNELS


def test_fetch_with_optional_modalities(tmp_path):
    """Ensure optional LiDAR and hyperspectral sources are stacked correctly."""
    lidar = np.random.rand(32, 32).astype(np.float32)
    lidar_path = tmp_path / "lidar.npy"
    np.save(lidar_path, lidar)

    spectral = np.random.rand(4, 32, 32).astype(np.float32)
    spectral_path = tmp_path / "spectral.npy"
    np.save(spectral_path, spectral)

    data = fetch_satellite_image(
        0.0,
        0.0,
        size=16,
        lidar_path=str(lidar_path),
        spectral_path=str(spectral_path),
    )

    assert data.shape == (NUM_CHANNELS + 1 + 4, 16, 16)
    assert data.min() >= 0.0 and data.max() <= 1.0


