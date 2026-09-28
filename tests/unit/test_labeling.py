"""Unit tests for detection caching, prompt management, and zero-shot detectors."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from sivia.labeling.cache import DetectionCache
from sivia.labeling.grounding_dino import GroundingDinoDetector
from sivia.labeling.owlv2 import Owlv2Detector
from sivia.labeling.prompts import PromptManager
from sivia.labeling.types import RawDetection


def test_detection_cache(tmp_path: Path) -> None:
    cache = DetectionCache(cache_dir=tmp_path / "cache")
    key = DetectionCache.compute_key("fake_sha", "owlv2", {"0": "screwdriver"})

    assert cache.get(key) is None

    dets = [
        RawDetection(
            class_id=0, class_name="screwdriver", box_xyxy=[10.0, 10.0, 30.0, 30.0], score=0.9
        )
    ]
    cache.put(key, dets)

    loaded = cache.get(key)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    assert loaded[0].box_xyxy == [10.0, 10.0, 30.0, 30.0]
    assert loaded[0].score == 0.9


def test_prompt_manager() -> None:
    mgr = PromptManager("configs/labeling/prompts.yaml")
    sets = mgr.get_prompt_sets()
    assert "default" in sets
    assert "descriptive" in sets

    class_names = mgr.get_class_names()
    assert class_names[0] == "screwdriver"
    assert class_names[1] == "tape_roll"

    prompts = mgr.get_prompts_for_set("default")
    assert 0 in prompts
    assert "screwdriver" in prompts[0]


def test_mock_detectors() -> None:
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    class_prompts = {0: "screwdriver", 1: "tape_roll"}

    owl = Owlv2Detector(mock=True)
    o_dets = owl.detect(img, class_prompts=class_prompts, use_cache=False)
    assert len(o_dets) == 2
    for d in o_dets:
        assert 0 <= d.box_xyxy[0] < d.box_xyxy[2] <= 200
        assert 0 <= d.box_xyxy[1] < d.box_xyxy[3] <= 200

    gdino = GroundingDinoDetector(mock=True)
    g_dets = gdino.detect(img, class_prompts=class_prompts, use_cache=False)
    assert len(g_dets) == 2
    for d in g_dets:
        assert 0 <= d.box_xyxy[0] < d.box_xyxy[2] <= 200
        assert 0 <= d.box_xyxy[1] < d.box_xyxy[3] <= 200
