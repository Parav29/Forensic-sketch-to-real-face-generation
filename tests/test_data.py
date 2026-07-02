import numpy as np
from PIL import Image

from data.preprocess import subject_id, split_indices
from data.alignment import align_face
from data.dataset import SketchPhotoDataset
from sketch_styles import available_styles, apply_style


def test_subject_id_strips_variants():
    assert subject_id("m-001-01_sketch") == "m-001"
    assert subject_id("00042_a") == "00042"
    assert subject_id("plainname") == "plainname"


def test_split_is_subject_aware_no_leakage():
    stems = [f"subj{s}_{v:02d}" for s in range(10) for v in range(3)]
    assign = split_indices(stems, subject_aware=True, seed=0)
    # Every image of a subject must land in the same split.
    by_subject = {}
    for st in stems:
        by_subject.setdefault(subject_id(st), set()).add(assign[st])
    assert all(len(splits) == 1 for splits in by_subject.values())


def test_alignment_center_fallback_shape():
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    out = align_face(img, 256, backend="center")
    assert out.shape == (256, 256, 3)


def test_sketch_styles_registry():
    styles = available_styles()
    assert {"xdog", "canny", "pencil", "adaptive"}.issubset(set(styles))
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    for s in styles:
        out = apply_style(s, img)
        assert out.shape == (128, 128) and out.dtype == np.uint8


def test_dataset_item(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    pair = (np.random.rand(256, 512, 3) * 255).astype(np.uint8)
    Image.fromarray(pair).save(d / "subj0_01.png")
    ds = SketchPhotoDataset(str(tmp_path), "train", augment=False)
    item = ds[0]
    assert item["sketch"].shape == (3, 256, 256)
    assert item["photo"].shape == (3, 256, 256)
    assert item["filename"] == "subj0_01"
