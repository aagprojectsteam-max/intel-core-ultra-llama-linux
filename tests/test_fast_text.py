from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
from test_aag_llama_models import gguf
FIXTURES = tempfile.TemporaryDirectory()
ROOT = Path(FIXTURES.name)
QAT = str(gguf(ROOT / 'gemma-4-12b-it-qat-q4_0.gguf', {'general.architecture': 'gemma4', 'gemma4.embedding_length': 3840, 'gemma4.context_length': 32768, 'general.file_type': 2}))
HAU = str(gguf(ROOT / 'Gemma4-12B-HauhauCS-Q4_K_M.gguf', {'general.architecture': 'gemma4', 'gemma4.embedding_length': 3840, 'gemma4.context_length': 32768}))
MTP = str(ROOT / 'mtp-gemma-4-12B-it.gguf')
Path(MTP).touch()
PROJECTOR = str(ROOT / 'mmproj-gemma-4-12b-it-qat-q4_0.gguf')
LAUNCHER = HERE.parent / 'bin/aag-llama-control'
SHELL = '''source "$1"
TEST_MODEL="$2"; TEST_MTP="$3"; TEST_PROJECTOR="$4"; FAST_TEXT_TARGET="$5"; FAST_TEXT_MTP="$6"; FAST_TEXT_MTP3=1; shift 6
validate_common_paths() { :; }
load_oneapi() { :; }
prepare_start() { return 99; }
choose_model() { printf '%s' "${1:-$TEST_MODEL}"; }
choose_companion() { [[ "$3" == off ]] && return 0; if [[ "$2" == mtp ]]; then printf '%s' "$TEST_MTP"; else printf '%s' "$TEST_PROJECTOR"; fi; }
server_help_stub() { printf '%s\\n' '--spec-draft-model draft-mtp'; }
LLAMA_SERVER=server_help_stub
start_server --dry-run "$@"
'''


def invoke(launcher=LAUNCHER, model=QAT, mtp='', projector='', extra=()):
    return subprocess.run(['bash', '-c', SHELL, 'test', str(launcher), model, mtp, projector, QAT, MTP, *extra], capture_output=True, text=True)


def command(**kwargs):
    result = invoke(**kwargs)
    if result.returncode:
        raise AssertionError(result.stderr)
    return shlex.split(result.stdout.split('COMMAND:', 1)[1])


class FastTextTest(unittest.TestCase):
    def assert_profile(self, args):
        for key, value in {'--ctx-size': '32768', '--batch-size': '512', '--ubatch-size': '512', '--threads': '8', '--threads-batch': '8', '--flash-attn': 'on', '--cache-type-k': 'f16', '--cache-type-v': 'f16', '--n-gpu-layers': 'all', '--port': '8080', '--parallel': '1', '--reasoning': 'off'}.items():
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--jinja', args)
        self.assertNotIn('--fit', args)

    def test_no_mtp_profile(self):
        args = command()
        self.assert_profile(args)
        self.assertNotIn('--spec-draft-model', args)

    def test_mtp_profile(self):
        args = command(mtp=MTP)
        self.assert_profile(args)
        for key, value in {'--spec-draft-model': MTP, '--spec-type': 'draft-mtp', '--spec-draft-n-max': '3', '--spec-draft-n-min': '0', '--spec-draft-p-min': '0'}.items():
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--spec-draft-backend-sampling', args)

    def test_explicit_fast_text(self):
        self.assert_profile(command(projector=PROJECTOR, extra=('--profile', 'fast-text')))
        self.assertNotIn('--mmproj', command(projector=PROJECTOR, extra=('--profile', 'fast-text')))
        rejected = invoke(extra=('--profile', 'fast-text', '--model', HAU))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('requires the configured model', rejected.stderr)
        self.assertNotEqual(invoke(extra=('--profile', 'fast-text', '--mmproj', 'on')).returncode, 0)

    def test_other_models_and_vision_unchanged(self):
        for model, mtp, projector in [(HAU, '', ''), (HAU, MTP, ''), (QAT, MTP, PROJECTOR), (QAT, '', PROJECTOR)]:
            with self.subTest(model=model, mtp=mtp, projector=projector):
                args = command(model=model, mtp=mtp, projector=projector)
                self.assertEqual(args[args.index('--model') + 1], model)
                self.assertEqual(args[args.index('--ctx-size') + 1], '32768')
                self.assertEqual(args[args.index('--port') + 1], '8080')
                self.assertEqual(args[args.index('--ubatch-size') + 1], '128')
                self.assertNotIn('--spec-draft-backend-sampling', args)
                if mtp:
                    self.assertEqual(args[args.index('--spec-draft-n-max') + 1], '2')
                if projector:
                    self.assertEqual(args[args.index('--mmproj') + 1], projector)
                else:
                    self.assertNotIn('--mmproj', args)

    def test_existing_explicit_context_choices(self):
        for profile, expected in [('fast', '4096'), ('normal', '8192'), ('long', '32768')]:
            args = command(extra=('--profile', profile))
            self.assertEqual(args[args.index('--ctx-size') + 1], expected)

    def test_unknown_companion_not_given_mtp3(self):
        args = command(mtp='/tmp/unrelated-mtp.gguf')
        self.assertEqual(args[args.index('--spec-draft-n-max') + 1], '2')

    def test_canonical_alias_and_menu(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / Path(QAT).name
            alias.symlink_to(QAT)
            self.assert_profile(command(model=str(alias)))
            script = 'source "$1"; QAT_TEST="$2"; HAU_TEST="$3"; FAST_TEXT_TARGET="$(readlink -f "$2")"; list_server_models() { printf \'%s\\0\' "$QAT_TEST" "$HAU_TEST"; }; choose_model "" server'
            result = subprocess.run(['bash', '-c', script, 'test', str(LAUNCHER), str(alias), HAU], input='1\n', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('[FAST TEXT]', result.stderr)
            self.assertIn(Path(HAU).name, result.stderr)
            self.assertIn('V = Vision  T = Tools  R = Reasoning  M =', result.stderr)

    def test_fixture_metadata_and_opt_in_prompt(self):
        from aag_llama_models import capabilities, matching_companions
        from test_aag_llama_models import gguf
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            model = gguf(fixture / Path(QAT).name, {'general.architecture': 'gemma4', 'general.name': 'Gemma4-12B', 'gemma4.embedding_length': 3840, 'tokenizer.ggml.tokens': ['a', 'b']})
            gguf(fixture / Path(MTP).name, {'general.architecture': 'gemma4-assistant', 'general.name': 'Gemma4-12B', 'gemma4-assistant.embedding_length_out': 3840, 'gemma4-assistant.nextn_predict_layers': 4, 'tokenizer.ggml.tokens': ['a', 'b']})
            gguf(fixture / Path(PROJECTOR).name, {'general.architecture': 'clip', 'general.name': 'Gemma4-12B', 'clip.has_vision_encoder': True, 'clip.vision.projection_dim': 3840})
            self.assertTrue(capabilities(model)['mtp'])
            self.assertTrue(capabilities(model)['vision'])
            self.assertEqual(len(matching_companions(model, 'mtp')), 1)
            for answer, expected in [('y\n', str(fixture / Path(MTP).name)), ('n\n', ''), ('\n', '')]:
                result = subprocess.run(['bash', '-c', 'source "$1"; choose_companion "$2" mtp ask', 'test', str(LAUNCHER), str(model)], input=answer, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                self.assertIn('Load with MTP (experimental)? [y/N]', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
