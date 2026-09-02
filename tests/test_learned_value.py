import numpy as np

from scheduler.learned_value import PATH_VALUE_FEATURES, SmallPathValueModel


def test_small_path_value_model_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(4)
    features = rng.normal(size=(80, len(PATH_VALUE_FEATURES))).astype(np.float32)
    targets = (0.7 * features[:, 0] - 0.4 * features[:, 1] + 0.2).astype(
        np.float32
    )
    model = SmallPathValueModel(hidden_size=8, seed=3).fit(
        features, targets, epochs=1000
    )
    prediction = model.predict(features)
    assert np.mean((prediction - targets) ** 2) < 0.03

    path = tmp_path / "value_model.npz"
    model.save(path)
    loaded = SmallPathValueModel.load(path)
    assert np.allclose(loaded.predict(features[:5]), prediction[:5])

