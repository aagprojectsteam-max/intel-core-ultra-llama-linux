from pathlib import Path
import struct
import tempfile
import unittest

from aag_llama_models import discover, matching_companions, metadata
from gguf_metadata import read_header
from test_aag_llama_models import gguf


class MetadataSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metadata.cache_clear()

    def test_malformed_headers_are_bounded(self):
        for data in [b"", b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 2**63), b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 2**63)]:
            path = self.root / "bad.gguf"
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                read_header(path)

    def test_discovery_excludes_sidecars_and_secondary_shards(self):
        fields = {"general.architecture": "qwen35", "qwen35.context_length": 4096}
        good = gguf(self.root / "model-00001-of-00002.gguf", fields)
        for name in ["model-00002-of-00002.gguf", "mtp-model.gguf", "model-mtp-q8.gguf", "mmproj-model.gguf", "dflash-model.gguf", "bad\nname.gguf"]:
            gguf(self.root / name, fields)
        gguf(self.root / "assistant.gguf", {"general.architecture": "gemma4-assistant"})
        (self.root / "bad.gguf").write_bytes(b"not a model")
        self.assertEqual(list(discover(self.root)), [str(good)])

    def test_discovery_does_not_follow_external_symlinks(self):
        inner = self.root / "inner"
        inner.mkdir()
        outside = gguf(self.root / "outside.gguf", {"general.architecture": "qwen35"})
        (inner / "linked.gguf").symlink_to(outside)
        (inner / "cycle").symlink_to(inner, target_is_directory=True)
        self.assertEqual(list(discover(inner)), [])

    def test_projector_without_dimensions_is_not_confident(self):
        model = gguf(self.root / "model.gguf", {"general.architecture": "qwen35", "general.name": "model"})
        gguf(self.root / "mmproj-model.gguf", {"general.architecture": "clip", "general.name": "model", "clip.has_vision_encoder": True})
        self.assertEqual(matching_companions(model, "mmproj"), [])

    def test_invalid_string_metadata_does_not_become_a_capability(self):
        model = gguf(self.root / "invalid-type.gguf", {"general.architecture": 42, "general.name": True, "tokenizer.chat_template": 23})
        self.assertEqual(list(discover(self.root)), [])
        self.assertEqual(metadata(str(model)), {})
