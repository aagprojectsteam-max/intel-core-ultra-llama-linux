from pathlib import Path
import json
import shlex
import subprocess
import tempfile
import unittest

from aag_llama_models import capabilities, launcher_settings, metadata
from test_aag_llama_models import gguf

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE.parent / 'bin/aag-llama-control'
SHELL = '''source "$1"
MODEL_ROOT="$2"
FAST_TEXT_TARGET="$3"
FAST_TEXT_MTP="$4"
FAST_TEXT_MTP3=1
EXPECTED_BIN=/bin/true
shift 4
load_oneapi() { :; }
prepare_start() { echo UNEXPECTED_STATE_CHANGE; return 99; }
server_help_stub() { printf '%s\\n' '--spec-draft-model draft-mtp --reasoning --chat-template-kwargs'; }
LLAMA_SERVER=server_help_stub
start_server --dry-run "$@"
'''


class ServerFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'Qwen3.5-4B-Q4_K_M.gguf'
        self.fields = {'general.architecture': 'qwen35', 'general.name': 'Qwen3.5-4B', 'qwen35.context_length': 32768, 'qwen35.embedding_length': 2048, 'tokenizer.ggml.tokens': ['a', 'b'], 'tokenizer.chat_template': '{% if tools %}tool_calls{% endif %}{% if enable_thinking %}<think></think>{% endif %}'}
        self.rewrite()
        self.fast_target = '/not-the-official-target'
        self.approved_draft = '/not-the-approved-draft'

    def rewrite(self):
        gguf(self.model, self.fields)
        metadata.cache_clear()

    def companion(self, kind, alternate=False):
        name = kind + ('-alternate' if alternate else '-Qwen3.5-4B') + '.gguf'
        if kind == 'mmproj':
            fields = {'general.architecture': 'clip', 'general.name': 'Qwen3.5-4B', 'clip.has_vision_encoder': True, 'clip.vision.projection_dim': 2048}
        else:
            fields = {'general.architecture': 'qwen35', 'general.name': 'Qwen3.5-4B', 'qwen35.embedding_length': 2048, 'qwen35.nextn_predict_layers': 1, 'tokenizer.ggml.tokens': ['a', 'b']}
        return gguf(self.root / name, fields)

    def invoke(self, answers='', extra=(), wizard=True, select_model=False):
        args = ['bash', '-c', SHELL, 'test', str(LAUNCHER), str(self.root), str(self.fast_target), str(self.approved_draft)]
        if wizard:
            args += ['--interactive']
        if not select_model:
            args += ['--model', str(self.model)]
        result = subprocess.run(args + list(extra), input=answers, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        self.assertNotIn('UNEXPECTED_STATE_CHANGE', result.stdout)
        return result

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('DRY RUN: no server or service changes', result.stdout)
        return shlex.split(result.stdout.split('COMMAND:', 1)[1])

    def value(self, args, flag):
        self.assertEqual(args.count(flag), 1, args)
        return args[args.index(flag) + 1]

    def test_no_companions_and_no_reasoning(self):
        self.fields['tokenizer.chat_template'] = '{{ messages }}'
        self.rewrite()
        result = self.invoke('3\n2\ny\n')
        args = self.command(result)
        self.assertNotIn('Vision/mmproj:', result.stdout)
        self.assertNotIn('MTP:', result.stdout)
        self.assertNotIn('Reasoning:', result.stdout)
        self.assertEqual(self.value(args, '--ctx-size'), '8192')
        self.assertNotIn('--mmproj', args)
        self.assertNotIn('--spec-draft-model', args)

    def test_full_order_and_badges(self):
        projector = self.companion('mmproj')
        draft = self.companion('mtp')
        result = self.invoke('1\n1\n1\n2\n3\n1\ny\n', select_model=True)
        args = self.command(result)
        markers = ['Select model number:', 'Vision/mmproj:', '\nMTP:', '\nReasoning:', '\nContext:', '\nPerformance:', 'Model:        ', 'Start server? [Y/n]:', 'COMMAND:']
        positions = [result.stdout.index(x) for x in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('[V][T][R][M]', result.stdout)
        self.assertIn('V = Vision  T = Tools  R = Reasoning  M =', result.stdout)
        self.assertEqual(self.value(args, '--mmproj'), str(projector))
        self.assertEqual(self.value(args, '--spec-draft-model'), str(draft))
        self.assertEqual(self.value(args, '--reasoning'), 'on')
        self.assertEqual(self.value(args, '--ctx-size'), '8192')
        self.assertIn('Backend:      SYCL', result.stdout)
        self.assertIn('GGML_SYCL_ENABLE_GRAPH=0', args)
        self.assertIn('GGML_SYCL_ENABLE_DNN=1', args)

    def test_one_projector_model_only(self):
        self.companion('mmproj')
        args = self.command(self.invoke('2\n1\n2\n2\ny\n'))
        self.assertNotIn('--mmproj', args)

    def test_multiple_projectors_selection_and_model_only(self):
        first = self.companion('mmproj')
        second = self.companion('mmproj', True)
        result = self.invoke('2\n1\n2\n2\ny\n')
        args = self.command(result)
        self.assertIn('0) Model only', result.stdout)
        self.assertEqual(self.value(args, '--mmproj'), str(sorted([first, second])[1]))
        self.assertNotIn('--mmproj', self.command(self.invoke('0\n1\n2\n2\ny\n')))

    def test_mtp_on_off_and_multiple(self):
        self.companion('mtp')
        args = self.command(self.invoke('1\n1\n2\n2\ny\n'))
        self.assertEqual(self.value(args, '--spec-type'), 'draft-mtp')
        self.assertEqual(self.value(args, '--spec-draft-n-max'), '2')
        self.assertNotIn('--spec-draft-model', self.command(self.invoke('2\n1\n2\n2\ny\n')))
        self.companion('mtp', True)
        self.assertIn('--spec-draft-model', self.command(self.invoke('2\n1\n2\n2\ny\n')))

    def test_ambiguous_noninteractive_on_does_not_pick_arbitrarily(self):
        self.companion('mmproj')
        self.companion('mmproj', True)
        result = self.invoke(extra=('--mmproj', 'on'), wizard=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('COMMAND:', result.stdout)

    def test_toggle_reasoning_rejects_invented_levels(self):
        args = self.command(self.invoke('2\n2\n2\ny\n'))
        self.assertEqual(self.value(args, '--reasoning'), 'on')
        result = self.invoke(extra=('--reasoning', 'medium'), wizard=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not supported', result.stdout)

    def test_reasoning_badge_without_control(self):
        self.fields['tokenizer.chat_template'] = '<think></think>{{ messages }}'
        self.rewrite()
        self.assertTrue(capabilities(self.model)['reasoning'])
        self.assertEqual(launcher_settings(self.model)[1], 'inherent')
        result = self.invoke('2\n2\ny\n')
        args = self.command(result)
        self.assertIn('Reasoning: inherent; no verified user control', result.stdout)
        self.assertNotIn('\n1) OFF', result.stdout)
        self.assertEqual(self.value(args, '--reasoning'), 'auto')

    def test_reasoning_mentions_are_not_controls(self):
        for template in ['{# {% if enable_thinking %} #}<think></think>', '{% set enable_thinking = true %}{% if enable_thinking %}<think></think>{% endif %}']:
            self.fields['tokenizer.chat_template'] = template
            self.rewrite()
            self.assertEqual(launcher_settings(self.model)[1], 'inherent')

    def test_gpt_oss_supported_effort_levels(self):
        self.fields.update({'general.architecture': 'gpt-oss', 'gpt-oss.context_length': 131072, 'tokenizer.chat_template': '{{ "Reasoning: " + reasoning_effort }}<|channel|>analysis'})
        self.rewrite()
        for number, level in enumerate(['low', 'medium', 'high'], 1):
            result = self.invoke(f'{number}\n2\n2\ny\n')
            args = self.command(result)
            self.assertNotIn('1) OFF', result.stdout)
            self.assertEqual(self.value(args, '--reasoning'), 'on')
            self.assertEqual(json.loads(self.value(args, '--chat-template-kwargs')), {'reasoning_effort': level})
        self.assertNotEqual(self.invoke(extra=('--reasoning', 'off'), wizard=False).returncode, 0)

    def test_granite_template_thinking(self):
        self.fields.update({'general.architecture': 'granite', 'granite.context_length': 131072, 'tokenizer.chat_template': '{% if thinking %}<think></think>{% endif %}'})
        self.rewrite()
        for number, enabled in [(1, False), (2, True)]:
            result = self.invoke(f'{number}\n2\n2\ny\n')
            args = self.command(result)
            self.assertEqual(json.loads(self.value(args, '--chat-template-kwargs')), {'thinking': enabled})
            self.assertIn('without client system messages, tools, or documents', result.stdout)

    def test_context_menu_and_metadata_limit(self):
        self.fields['qwen35.context_length'] = 8192
        self.rewrite()
        result = self.invoke('1\n3\n2\ny\n')
        args = self.command(result)
        self.assertEqual(self.value(args, '--ctx-size'), '8192')
        self.assertNotIn('4) 16384', result.stdout)
        self.assertNotIn('5) 32768', result.stdout)
        self.assertIn('Server capacity, not the size of every prompt', result.stdout)
        for invalid in ['0', '-1', '8193', 'text', '99999999999999999999', '1+4096']:
            result = self.invoke(extra=('--context', invalid), wizard=False)
            self.assertNotEqual(result.returncode, 0, invalid)
            self.assertNotIn('COMMAND:', result.stdout)

    def test_custom_context_retry(self):
        result = self.invoke('1\n6\n999999999999999999999999\n6\n-1\n6\n12000\n2\ny\n')
        args = self.command(result)
        self.assertEqual(result.stdout.count('Invalid context;'), 2)
        self.assertEqual(self.value(args, '--ctx-size'), '12000')
        args = self.command(self.invoke(extra=('--context', '0008192'), wizard=False))
        self.assertEqual(self.value(args, '--ctx-size'), '8192')

    def test_unknown_context_does_not_invent_a_limit(self):
        self.fields.pop('qwen35.context_length')
        self.rewrite()
        result = self.invoke('1\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Cannot establish', result.stdout)

    def test_performance_profiles(self):
        self.fields.update({'general.architecture': 'gemma4', 'gemma4.context_length': 262144, 'gemma4.embedding_length': 3840})
        self.rewrite()
        for number, name, ubatch, threads in [(1, 'FAST', '512', '8'), (2, 'MEDIUM', '128', '8'), (3, 'SLOW', '128', '4')]:
            result = self.invoke(f'1\n3\n{number}\ny\n')
            args = self.command(result)
            self.assertIn(f'Performance:  {name}', result.stdout)
            for key, value in {'--batch-size': '512', '--ubatch-size': ubatch, '--threads': threads, '--threads-batch': threads, '--cache-type-k': 'f16', '--cache-type-v': 'f16', '--n-gpu-layers': 'all', '--flash-attn': 'on'}.items():
                self.assertEqual(self.value(args, key), value)

    def test_explicit_performance_does_not_change_context(self):
        for mode in ['fast', 'medium', 'slow', 'conservative']:
            args = self.command(self.invoke(extra=('--context', '4096', '--performance', mode), wizard=False))
            self.assertEqual(self.value(args, '--ctx-size'), '4096')
            self.assertNotIn('Start server?', ' '.join(args))

    def test_fast_text_opt_in_mtp3_and_custom_capacity(self):
        self.fields.update({'general.architecture': 'gemma4', 'general.name': 'Gemma4-12B', 'gemma4.context_length': 32768, 'gemma4.embedding_length': 3840, 'general.file_type': 2})
        self.rewrite()
        self.fast_target = str(self.model)
        self.approved_draft = str(gguf(self.root / 'mtp-reference.gguf', {'general.architecture': 'gemma4-assistant', 'general.name': 'Gemma4-12B', 'gemma4-assistant.embedding_length_out': 3840, 'gemma4-assistant.nextn_predict_layers': 4, 'tokenizer.ggml.tokens': ['a','b']}))
        args = self.command(self.invoke(extra=('--profile', 'fast-text', '--mtp', 'on'), wizard=False))
        for key, value in {'--ctx-size': '32768', '--ubatch-size': '512', '--spec-draft-n-max': '3', '--spec-draft-n-min': '0', '--spec-draft-p-min': '0', '--reasoning': 'off'}.items():
            self.assertEqual(self.value(args, key), value)
        self.assertIn('--spec-draft-backend-sampling', args)
        args = self.command(self.invoke('1\n1\n3\n1\ny\n', extra=('--profile', 'fast-text')))
        self.assertEqual(self.value(args, '--ctx-size'), '8192')
        self.assertEqual(self.value(args, '--spec-draft-n-max'), '3')

    def test_unattended_defaults_and_legacy_profiles(self):
        self.companion('mmproj')
        self.companion('mtp')
        for profile, context in [('', '32768'), ('fast', '4096'), ('normal', '8192'), ('long', '32768')]:
            result = self.invoke(extra=('--profile', profile) if profile else (), wizard=False)
            args = self.command(result)
            self.assertEqual(self.value(args, '--ctx-size'), context)
            self.assertEqual(self.value(args, '--reasoning'), 'off')
            self.assertNotIn('--mmproj', args)
            self.assertNotIn('--spec-draft-model', args)
            for text in ['Select [', 'Reasoning:', 'Context:', 'Performance:', 'Start server?']:
                self.assertNotIn(text, result.stdout)
        result = self.invoke('y\n', extra=('--mtp', 'ask', '--mmproj', 'off'), wizard=False)
        self.assertIn('--spec-draft-model', self.command(result))

    def test_summary_decline_and_eof_never_launch(self):
        for ending in ['n\n', '']:
            result = self.invoke('1\n3\n2\n' + ending)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn('Cancelled.', result.stdout)
            self.assertNotIn('COMMAND:', result.stdout)
        result = self.invoke('')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('COMMAND:', result.stdout)

    def test_generic_fast_text_mapping_does_not_inherit_mtp3(self):
        self.fast_target = str(self.model)
        self.approved_draft = str(self.companion('mtp'))
        args = self.command(self.invoke(extra=('--profile', 'fast-text', '--mtp', 'on'), wizard=False))
        self.assertEqual(self.value(args, '--ubatch-size'), '128')
        self.assertEqual(self.value(args, '--spec-draft-n-max'), '2')
        self.assertNotIn('--spec-draft-backend-sampling', args)

    def test_unattended_context_is_clamped_to_metadata(self):
        self.fields['qwen35.context_length'] = 4096
        self.rewrite()
        args = self.command(self.invoke(extra=('--profile', 'long'), wizard=False))
        self.assertEqual(self.value(args, '--ctx-size'), '4096')
        self.assertIn('GGML_SYCL_ENABLE_GRAPH=0', args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
