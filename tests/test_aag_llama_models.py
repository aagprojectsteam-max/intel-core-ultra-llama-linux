from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from aag_llama_models import capabilities, matching_companions, metadata, normalized


LAUNCHER = Path(__file__).resolve().parents[1] / "bin/aag-llama-control"
TEMPLATE = "{% if tools %}{% for tool in tools %}{{ tool }}{% endfor %}{% endif %} tool_calls <think></think>"


def gguf(path, fields, tensor=None):
    def string(value):
        value = value.encode()
        return struct.pack("<Q", len(value)) + value
    data = b"GGUF" + struct.pack("<IQQ", 3, int(tensor is not None), len(fields))
    for key, value in fields.items():
        data += string(key)
        if isinstance(value, bool):
            data += struct.pack("<I?", 7, value)
        elif isinstance(value, int):
            data += struct.pack("<II", 4, value)
        elif isinstance(value, list):
            data += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(string(item) for item in value)
        else:
            data += struct.pack("<I", 8) + string(value)
    if tensor:
        name, columns, rows = tensor
        data += string(name) + struct.pack("<IQQIQ", 2, columns, rows, 0, 0)
    data += b"\0" * (-len(data) % 32)
    if tensor:
        data += b"\0" * (columns * rows * 4)
    path.write_bytes(data)
    return path


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metadata.cache_clear()
        self.main_fields = {"general.architecture": "qwen35moe", "general.name": "Qwen3.6-35B-A3B", "qwen35moe.embedding_length": 2048, "tokenizer.chat_template": TEMPLATE, "tokenizer.ggml.tokens": ["a", "b"]}
        self.model = gguf(self.root / "Qwen3.6-35B-A3B-Q4_K_M.gguf", self.main_fields)

    def companion(self, kind, **changes):
        if kind == "mmproj":
            fields = {"general.architecture": "clip", "general.name": "Qwen3.6-35B-A3B", "clip.has_vision_encoder": True, "clip.vision.projection_dim": 2048}
            name = "mmproj-Qwen3.6-35B-A3B-Q8_0.gguf"
        else:
            fields = {"general.architecture": "qwen35moe", "general.name": "Qwen3.6-35B-A3B", "qwen35moe.embedding_length": 2048, "qwen35moe.nextn_predict_layers": 1, "tokenizer.ggml.tokens": ["a", "b"]}
            name = "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
        fields.update(changes)
        return gguf(self.root / name, fields)

    def shell(self, command, input=""):
        return subprocess.run(["bash", "-c", 'source "$1"; ' + command, "test", str(LAUNCHER)], text=True, input=input, capture_output=True)

    def test_native_header_reader(self):
        self.assertEqual(metadata(str(self.model))["general.name"], "Qwen3.6-35B-A3B")

    def test_tools_and_reasoning_from_template(self):
        result = capabilities(self.model)
        self.assertTrue(result["tools"])
        self.assertTrue(result["reasoning"])
        self.assertFalse(result["vision"])

    def test_gemma_thinking(self):
        self.main_fields["tokenizer.chat_template"] = "enable_thinking <|channel>thought"
        gguf(self.model, self.main_fields)
        self.assertTrue(capabilities(self.model)["reasoning"])

    def test_uncensored_gemma_thinking_control(self):
        self.main_fields["tokenizer.chat_template"] = "enable_thinking <|think|> <|channel>"
        gguf(self.model, self.main_fields)
        self.assertTrue(capabilities(self.model)["reasoning"])

    def test_gpt_oss_thinking(self):
        self.main_fields["tokenizer.chat_template"] = "<|channel|>analysis"
        gguf(self.model, self.main_fields)
        self.assertTrue(capabilities(self.model)["reasoning"])

    def test_explicit_negative_capabilities(self):
        self.main_fields.update({"aag.capabilities.tools": False, "aag.capabilities.reasoning": False})
        gguf(self.model, self.main_fields)
        self.assertEqual(capabilities(self.model)["badge"], "[ ][ ][ ][ ]")

    def test_vision_and_matching_mmproj(self):
        companion = self.companion("mmproj")
        self.assertEqual(matching_companions(self.model, "mmproj"), [str(companion)])
        self.assertTrue(capabilities(self.model)["vision"])

    def test_generic_projector_metadata(self):
        companion = self.companion("mmproj")
        generic = companion.with_name("mmproj-BF16.gguf")
        companion.rename(generic)
        self.assertEqual(matching_companions(self.model, "mmproj"), [str(generic)])

    def test_e2b_projector_with_numbered_precision_suffix(self):
        model = gguf(self.root / "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf", {"general.architecture": "gemma4", "general.name": "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive", "gemma4.embedding_length": 1536})
        projector = gguf(self.root / "mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16_2.gguf", {"general.architecture": "clip", "general.name": "KL0.1963 4Ref", "clip.has_vision_encoder": True, "clip.vision.projection_dim": 1536})
        self.assertEqual(matching_companions(model, "mmproj"), [str(projector)])

    def test_mtp_requires_trained_head(self):
        self.companion("mtp", **{"qwen35moe.nextn_predict_layers": 0})
        self.assertFalse(capabilities(self.model)["mtp"])

    def test_mtp_and_rendering(self):
        self.companion("mmproj")
        companion = self.companion("mtp")
        result = capabilities(self.model)
        self.assertEqual(result["mtp_candidates"], [str(companion)])
        self.assertEqual(result["badge"], "[V][T][R][M]")

    def test_incompatible_mtp_architecture(self):
        self.companion("mtp", **{"general.architecture": "qwen35"})
        self.assertEqual(matching_companions(self.model, "mtp"), [])

    def test_incompatible_projector_width(self):
        self.companion("mmproj", **{"clip.vision.projection_dim": 4096})
        self.assertEqual(matching_companions(self.model, "mmproj"), [])

    def test_pixtral_uses_actual_output_tensor_for_bad_metadata(self):
        model = gguf(self.root / "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", {"general.architecture": "mistral3", "mistral3.embedding_length": 3})
        projector = gguf(self.root / "Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf", {"general.architecture": "clip", "clip.has_vision_encoder": True, "clip.projector_type": "pixtral", "clip.vision.projection_dim": 9}, ("mm.2.weight", 2, 3))
        self.assertEqual(matching_companions(model, "mmproj"), [str(projector)])

    def test_pixtral_rejects_wrong_output_tensor(self):
        model = gguf(self.root / "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", {"general.architecture": "mistral3", "mistral3.embedding_length": 3})
        gguf(self.root / "Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf", {"general.architecture": "clip", "clip.has_vision_encoder": True, "clip.projector_type": "pixtral", "clip.vision.projection_dim": 9}, ("mm.2.weight", 2, 4))
        self.assertEqual(matching_companions(model, "mmproj"), [])

    def test_incompatible_mtp_width(self):
        self.companion("mtp", **{"qwen35moe.embedding_length": 4096})
        self.assertEqual(matching_companions(self.model, "mtp"), [])

    def test_incompatible_mtp_vocabulary(self):
        self.companion("mtp", **{"tokenizer.ggml.tokens": ["a"]})
        self.assertEqual(matching_companions(self.model, "mtp"), [])

    def test_missing_mtp_vocabulary(self):
        self.main_fields.pop("tokenizer.ggml.tokens")
        gguf(self.model, self.main_fields)
        gguf(self.root / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf", self.main_fields | {"qwen35moe.nextn_predict_layers": 1})
        self.assertEqual(matching_companions(self.model, "mtp"), [])

    def test_single_unrelated_companion_rejected(self):
        self.companion("mmproj", **{"general.name": "Unrelated"}).rename(self.root / "mmproj-unrelated.gguf")
        self.assertEqual(matching_companions(self.model, "mmproj"), [])

    def test_conflicting_metadata_overrides_matching_filename(self):
        self.companion("mmproj", **{"general.name": "Qwen3.5-35B-A3B"})
        self.assertEqual(matching_companions(self.model, "mmproj"), [])

    def test_explicit_disabled_companion(self):
        self.companion("mmproj")
        self.companion("mtp")
        self.main_fields.update({"aag.capabilities.vision": False, "aag.capabilities.mtp": False})
        gguf(self.model, self.main_fields)
        self.assertEqual(matching_companions(self.model, "mmproj"), [])
        self.assertEqual(matching_companions(self.model, "mtp"), [])

    def test_dotted_model_identity(self):
        self.assertNotEqual(normalized("Qwen3.5-4B"), normalized("Qwen3.6-35B-A3B"))

    def test_gemma_assistant_matching(self):
        model = gguf(self.root / "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf", {"general.architecture": "gemma4", "gemma4.embedding_length": 3840, "tokenizer.ggml.tokens": ["a", "b"]})
        draft = gguf(self.root / "mtp-gemma-4-12B-it.gguf", {"general.architecture": "gemma4-assistant", "gemma4-assistant.embedding_length": 1024, "gemma4-assistant.embedding_length_out": 3840, "gemma4-assistant.nextn_predict_layers": 4, "tokenizer.ggml.tokens": ["a", "b"]})
        self.assertEqual(matching_companions(model, "mtp"), [str(draft)])

    def test_no_companion(self):
        for kind in ["mmproj", "mtp"]:
            self.assertEqual(matching_companions(self.model, kind), [])

    def test_malformed_metadata_safe_fallback(self):
        self.model.write_bytes(b"not a GGUF")
        self.assertEqual(capabilities(self.model)["badge"], "[ ][ ][ ][ ]")

    def test_suggestive_filename_is_not_evidence(self):
        model = self.root / "Vision-Tools-Reasoning-MTP.gguf"
        gguf(model, {"general.architecture": "unknown"})
        self.assertEqual(capabilities(model)["badge"], "[ ][ ][ ][ ]")

    def test_generic_image_template_does_not_make_functiongemma_vision(self):
        gguf(self.model, {"general.architecture": "gemma3", "tokenizer.chat_template": "'image' <start_of_image>"})
        self.assertFalse(capabilities(self.model)["vision"])

    def test_ambiguity_off_and_selection(self):
        for kind in ["mmproj", "mtp"]:
            first = self.companion(kind)
            second = first.with_name(kind + "-alternate.gguf")
            second.write_bytes(first.read_bytes())
            self.assertEqual(len(matching_companions(self.model, kind)), 2)
            command = f'choose_companion "{self.model}" {kind} ask'
            result = self.shell(command, "0\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertIn("0) Model only", result.stderr)
            result = self.shell(command, "2\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(result.stdout, matching_companions(self.model, kind))

    def test_prompts_yes_no_and_eof(self):
        for kind in ["mmproj", "mtp"]:
            companion = self.companion(kind)
            command = f'choose_companion "{self.model}" {kind} ask'
            for answer in ["y\n", "n\n", ""]:
                result = self.shell(command, answer)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Load with", result.stderr)
                self.assertEqual(result.stdout, str(companion) if answer == "y\n" else "")

    def test_explicit_mtp_auto_retains_approved_policy(self):
        self.companion("mtp")
        result = self.shell(f'choose_companion "{self.model}" mtp auto')
        self.assertEqual(result.stdout, "")
        self.assertIn("approved pair", result.stderr)

    def test_mtp_auto_selects_only_the_approved_candidate(self):
        approved = self.companion("mtp")
        alternate = approved.with_name("mtp-alternate.gguf")
        alternate.write_bytes(approved.read_bytes())
        command = f'APPROVED_MTP_TARGET="{self.model}"; approved_mtp_for_model() {{ printf "%s" "{approved}"; }}; choose_companion "{self.model}" mtp auto'
        result = self.shell(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(approved))
        self.assertNotIn("Select mtp", result.stderr)

    def test_server_menu_legend_and_companions_excluded(self):
        self.companion("mmproj")
        self.companion("mtp")
        result = self.shell(f'MODEL_ROOT="{self.root}"; choose_model "" server', "1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[V][T][R][M]", result.stderr)
        self.assertIn("V = Vision  T = Tools  R = Reasoning", result.stderr)
        self.assertEqual(result.stdout, str(self.model))
        self.assertNotIn("2)", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
