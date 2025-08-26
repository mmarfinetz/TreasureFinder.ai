import io
import os
import numpy as np
import types
import tempfile
import torch
import torch.nn as nn


class FakeEE:
    class Projection:
        def __init__(self, code):
            self.code = code

    class _Geom:
        def __init__(self, lon, lat):
            self.lon = float(lon)
            self.lat = float(lat)
        def transform(self, proj, maxError):
            # Fake transform: scale degrees to meters-ish so x,y vary with lon/lat
            # Not accurate UTM; just distinct and proportional for testing
            self._x = self.lon * 100000.0
            self._y = self.lat * 110000.0
            return self
        class _Coords:
            def __init__(self, x, y):
                self._x = x
                self._y = y
            def getInfo(self):
                return [self._x, self._y]
        def coordinates(self):
            return FakeEE._Geom._Coords(self._x, self._y)

    class Geometry:
        @staticmethod
        def Point(lon, lat):
            return FakeEE._Geom(lon, lat)

    class data:
        @staticmethod
        def computePixels(request):
            w = int(request['grid']['dimensions']['width'])
            h = int(request['grid']['dimensions']['height'])
            tx = float(request['grid']['affineTransform']['translateX'])
            ty = float(request['grid']['affineTransform']['translateY'])
            # Deterministic pattern that varies with origin
            xs = np.linspace(0, 1, w, dtype=np.float32)
            ys = np.linspace(0, 1, h, dtype=np.float32)
            grid = (ys[:, None] + xs[None, :]) / 2.0
            bias = (np.sin(tx/1e5) + np.cos(ty/1e5)) * 0.1
            chans = []
            for c in range(8):
                chans.append((c + 1) * (grid + bias))
            arr = np.stack(chans, axis=2).astype(np.float32)  # (H,W,C)
            b = io.BytesIO()
            np.save(b, arr)
            return b.getvalue()


def test_ee_grid_variance_and_logging(monkeypatch, caplog):
    import importlib
    import treasure_hunter_module as thm

    # Patch EE
    monkeypatch.setattr(thm, 'ee', FakeEE)
    monkeypatch.setattr(thm, 'EE_AVAILABLE', True)

    # Sample points around a center
    center = (36.35, -111.88)
    pts = []
    for dy in [-0.02, -0.01, 0, 0.01]:
        for dx in [-0.02, -0.01, 0, 0.01]:
            pts.append((center[0] + dy, center[1] + dx))

    tiles = []
    with caplog.at_level('INFO'):
        for (lat, lon) in pts:
            tile = thm.fetch_satellite_image(lat, lon, size=64)
            assert tile.shape == (thm.NUM_CHANNELS, 64, 64)
            tiles.append(tile)
    # Check logs contain CRS notification
    assert any('EE grid: CRS=' in rec.message for rec in caplog.records)

    # Verify tiles vary pairwise and per-band variance is present
    for t in tiles:
        for b in range(t.shape[0]):
            assert np.nanstd(t[b]) > 1e-6
    # Pairwise differences
    for i in range(len(tiles) - 1):
        mad = np.mean(np.abs(tiles[i] - tiles[i+1]))
        assert mad > 1e-3


def test_cnn_weights_loaded_and_nonconstant_scores(monkeypatch):
    # Prepare a temporary trained CNN file (random init ok for variance)
    import importlib
    import treasure_hunter_module as thm
    if not thm.TORCH_AVAILABLE:
        return

    fd, tmp = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    try:
        model = thm.SatelliteAnomalyCNN()
        thm.save_trained_model(model, tmp)
        # Enforce production env for CNN
        monkeypatch.setenv('PRODUCTION_MODE', 'true')
        monkeypatch.setenv('CNN_WEIGHTS_PATH', tmp)

        importlib.reload(thm)
        # Mock fetch to produce varying inputs
        def fake_fetch(lat, lon, size=thm.IMAGE_SIZE, **_):
            xs = np.linspace(0, 1, size, dtype=np.float32)
            ys = np.linspace(0, 1, size, dtype=np.float32)
            grid = ys[:, None] + xs[None, :]
            bias = (np.sin(lat) + np.cos(lon)).astype(np.float32) if isinstance(lat, np.ndarray) else float(np.sin(lat) + np.cos(lon))
            chans = [((i+1) * (grid + bias)).astype(np.float32) for i in range(thm.NUM_CHANNELS)]
            return np.stack(chans, axis=0)

        monkeypatch.setattr(thm, 'fetch_satellite_image', fake_fetch)

        # Run N>10 points and check score variability
        scores = []
        for i in range(12):
            lat = 36.0 + 0.01 * i
            lon = -112.0 + 0.01 * (i // 2)
            res = thm.analyze_satellite_anomalies(lat, lon, use_dofa=False)
            scores.append(res['anomaly_score'])
        assert np.std(scores) > 0.05
        # Not all around 0.5
        assert not all(abs(s - 0.5) < 1e-3 for s in scores)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def test_dofa_production_requirements(monkeypatch):
    import importlib
    import treasure_hunter_module as thm
    if not thm.TORCH_AVAILABLE:
        return

    # Case 1: Missing local weights
    monkeypatch.setenv('PRODUCTION_MODE', 'true')
    monkeypatch.setenv('USE_DOFA', 'true')
    monkeypatch.delenv('DOFA_LOCAL_WEIGHTS', raising=False)
    importlib.reload(thm)
    with np.testing.assert_raises(RuntimeError) as ctx:
        thm.load_dofa_segmenter()
    assert 'DOFA_LOCAL_WEIGHTS' in str(ctx.exception)

    # Case 2: Plain state_dict should fail in production
    fd, tmp_sd = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    torch.save({'state_dict': {'foo.weight': torch.randn(1)}}, tmp_sd)
    try:
        monkeypatch.setenv('DOFA_LOCAL_WEIGHTS', tmp_sd)
        importlib.reload(thm)
        with np.testing.assert_raises(RuntimeError) as ctx2:
            thm.load_dofa_segmenter()
        assert 'serialized torch.nn.Module' in str(ctx2.exception) or 'convert your HF checkpoint' in str(ctx2.exception).lower()
    finally:
        os.remove(tmp_sd)

    # Case 3: Serialized Module succeeds
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 2, 1)
        def forward(self, x):
            return self.conv(x)

    fd, tmp_mod = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    # Use ScriptModule to avoid pickling local class
    scripted = torch.jit.script(Tiny())
    scripted.save(tmp_mod)
    try:
        monkeypatch.setenv('DOFA_LOCAL_WEIGHTS', tmp_mod)
        importlib.reload(thm)
        seg = thm.load_dofa_segmenter()
        assert seg is not None
    finally:
        os.remove(tmp_mod)
