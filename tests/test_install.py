from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

from test_aag_llama_models import gguf

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "intel-core-ultra-llama-linux"


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="intel-launcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "prefix with spaces"
        self.models = self.root / "models"
        self.models.mkdir()
        self.model = gguf(self.models / "Qwen3.5-4B-Q4_K_M.gguf", {
            "general.architecture": "qwen35", "general.name": "Qwen3.5-4B",
            "qwen35.context_length": 32768, "qwen35.embedding_length": 2048,
            "tokenizer.ggml.tokens": ["a", "b"],
            "tokenizer.chat_template": "{% if tools %}tool_calls{% endif %}{% if enable_thinking %}<think></think>{% endif %}"})
        self.build = self.root / "llama.cpp" / "build-sycl"
        self.stub = self.build / "bin"
        self.stub.mkdir(parents=True)
        self.executable(self.stub / "llama-server", "#!/bin/sh\nif [ \"$1\" = --help ]; then echo '--spec-draft-model draft-mtp --ctx-size'; exit 0; fi\nexit 98\n")
        self.executable(self.stub / "sycl-ls", "#!/bin/sh\necho '[level_zero:gpu:0] Intel(R) Arc(TM) Graphics'\n")
        self.oneapi = self.root / "setvars.sh"
        self.oneapi.write_text("export PATH=" + shlex.quote(str(self.stub)) + ':$PATH\n')
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("AAG_LLAMA_", "ONEAPI_", "SYCL_"))}
        self.env["PATH"] = "/usr/bin:/bin"
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["XDG_RUNTIME_DIR"] = str(self.root / "runtime")
        self.env["XDG_STATE_HOME"] = str(self.root / "state")
        Path(self.env["XDG_RUNTIME_DIR"]).mkdir()

    def executable(self, path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_command(self, args, input="", env=None):
        return subprocess.run([str(x) for x in args], input=input, text=True, capture_output=True, env=env or self.env, timeout=30)

    def install(self, extra=()):
        return self.run_command([ROOT / "install.sh", "--prefix", self.prefix, "--build-dir", self.build,
                                 "--llama-root", self.build.parent, "--model-root", self.models, "--oneapi-env", self.oneapi, *extra])

    def launcher(self, *args, input="", env=None):
        return self.run_command([self.prefix / "bin/aag-llama-server-start", *args], input, env)

    def uninstall(self):
        return self.run_command([self.prefix / "share" / PROJECT / "uninstall.sh", "--prefix", self.prefix])

    def test_clean_install_flow_and_uninstall(self):
        unrelated = self.prefix / "unrelated.txt"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("keep")
        self.assertEqual(self.install().returncode, 0)
        config = self.launcher("--show-config")
        self.assertEqual(config.returncode, 0, config.stderr)
        self.assertIn(str(self.models), config.stdout)
        self.assertIn(str(self.build), config.stdout)
        self.assertIn("1.0.0", self.launcher("--help").stdout)
        check = self.launcher("--check-runtime")
        self.assertEqual(check.returncode, 0, check.stderr)
        gguf(self.models / "mmproj-Qwen3.5-4B.gguf", {"general.architecture": "clip", "general.name": "Qwen3.5-4B", "clip.has_vision_encoder": True, "clip.vision.projection_dim": 2048})
        gguf(self.models / "mtp-Qwen3.5-4B.gguf", {"general.architecture": "qwen35", "general.name": "Qwen3.5-4B", "qwen35.embedding_length": 2048, "qwen35.nextn_predict_layers": 1, "tokenizer.ggml.tokens": ["a", "b"]})
        result = self.launcher("--interactive", "--dry-run", input="1\n1\n1\n2\n3\n1\ny\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stderr + result.stdout
        for marker in ("[V][T][R][M]", "Vision/mmproj:", "MTP:", "Reasoning:", "Context:", "Performance:", "Start server?", "COMMAND:"):
            self.assertIn(marker, text)
        self.assertIn("GGML_SYCL_ENABLE_GRAPH=0", text)
        self.assertNotIn("UNEXPECTED", text)
        self.assertEqual(self.uninstall().returncode, 0)
        self.assertFalse((self.prefix / "bin/aag-llama-server-start").exists())
        self.assertFalse((self.prefix / "share" / PROJECT / "bin/aag-llama-control").exists())
        self.assertTrue(self.model.exists())
        self.assertTrue((self.stub / "llama-server").exists())
        self.assertEqual(unrelated.read_text(), "keep")

    def test_collision_backup_upgrade_restore(self):
        target = self.prefix / "bin/aag-llama-server-start"
        target.parent.mkdir(parents=True)
        target.write_text("previous launcher\n")
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(target.read_text(), "previous launcher\n")
        result = self.install(("--replace",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.install().returncode, 0)
        self.assertEqual(self.uninstall().returncode, 0)
        self.assertEqual(target.read_text(), "previous launcher\n")

    def test_modified_config_and_command_are_preserved(self):
        self.assertEqual(self.install().returncode, 0)
        config = self.prefix / "config" / PROJECT / "config.conf"
        config.write_text("# user edit\n")
        command = self.prefix / "bin/aag-llama-cli-start"
        command.write_text("# user wrapper\n")
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.uninstall().returncode, 0)
        self.assertEqual(config.read_text(), "# user edit\n")
        self.assertEqual(command.read_text(), "# user wrapper\n")

    def test_missing_runtime_is_actionable(self):
        self.assertEqual(self.install().returncode, 0)
        result = self.launcher("--check-runtime", env=self.env | {"AAG_LLAMA_ONEAPI_ENV": str(self.root / "missing")})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("oneAPI/SYCL is missing", result.stderr)
        self.assertIn("INTEL-SYCL.md", result.stderr)

    def test_unattended_no_questions_and_capacity_clamp(self):
        self.assertEqual(self.install().returncode, 0)
        result = self.launcher("--model", self.model, "--dry-run", "--profile", "fast", "--mmproj", "off", "--mtp", "off")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ctx-size 4096", result.stdout)
        self.assertNotIn("Start server?", result.stdout)
        self.assertNotIn("Select", result.stderr)

    def test_symlink_prefix_is_rejected(self):
        destination = self.root / "unrelated"
        destination.mkdir()
        self.prefix.symlink_to(destination, target_is_directory=True)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(destination.iterdir()), [])
