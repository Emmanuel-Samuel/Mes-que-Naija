import numpy as np
import torch

from possession.seq_models import Prep, TCNNet, GRUNet, predict_logp_seq, train_seq


def test_prep_fills_nans_and_adds_indicators():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0], [np.nan, 6.0]] * 5, dtype=np.float32)
    p = Prep().fit([X], every=1)
    Z = p.transform(X)
    assert Z.shape == (20, 4) and not np.isnan(Z).any()
    assert Z[:, 2:].sum() == 10   # one NaN in each of the two columns, five times each


def test_windowed_prediction_covers_every_frame_and_is_normalised():
    torch.manual_seed(0)
    Z = np.random.RandomState(0).randn(3003, 5).astype(np.float32)
    for net in (GRUNet(5, 8), TCNNet(5, 8)):
        lp = predict_logp_seq(net, Z, window=100)
        assert lp.shape == (3003, 3) and np.isfinite(lp).all()
        assert np.allclose(np.exp(lp).sum(1), 1.0, atol=0.05)


def test_training_is_repeatable():
    rng = np.random.RandomState(1)
    Z = rng.randn(2000, 4).astype(np.float32)
    y = (Z[:, 0] > 0).astype(np.int64)
    cfg = dict(hidden=8, lr=5e-3, window=50, batch=4, steps=3, epochs=2, patience=2)
    outs = [predict_logp_seq(train_seq("gru", [(Z, y)], [(Z, y)], cfg, seed=0), Z, 50)
            for _ in range(2)]
    assert np.array_equal(outs[0], outs[1])
