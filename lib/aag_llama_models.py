#!/usr/bin/env python3
"""Read model capabilities without loading weights or starting a backend."""

import argparse
import functools
import json
import os
import re
import sys
from pathlib import Path

from gguf_metadata import read_header


def tensor_rows(path, name):
    try:
        return read_header(path, name)
    except (OSError, ValueError, UnicodeError):
        return None


@functools.lru_cache(maxsize=128)
def metadata(path):
    try:
        result = read_header(path)
        for key in ("general.architecture", "general.name", "tokenizer.chat_template", "clip.projector_type"):
            if key in result and not isinstance(result[key], str):
                result.pop(key)
        return result
    except (OSError, ValueError, UnicodeError) as error:
        print(f"Metadata unavailable for {Path(path).name}: {error}", file=sys.stderr)
        return {}


def normalized(value):
    value = Path(value).name.lower()
    if value.endswith(".gguf"):
        value = value[:-5]
    value = re.sub(r"(?:^|[-_.])(mmproj|mtp|dflash)(?:[-_.]|$)", "-", value)
    value = re.sub(r"(?:^|[-_.])(?:i?q\d+(?:[_-][a-z0-9]+)*|(?:bf16|fp16|f16)(?:_\d+)?)(?:$|[-_.])", "-", value)
    return re.sub(r"[^a-z0-9]", "", value)


def sidecar_kind(path):
    name = Path(path).stem.lower()
    for kind in ("mmproj", "mtp", "dflash"):
        if re.search(r"(?:^|[-_])" + kind + r"(?:[-_]|$)", name):
            return kind
    return None


def identity_matches(model, candidate, model_meta, candidate_meta):
    # Names establish identity; dimensions alone cannot establish compatibility.
    family = r"(?:qwen\d+(?:\.\d+)?|gemma[-_ ]?\d+)[-_ ]+(?:e?\d+b)(?:[-_ ]+a\d+b)?"
    main_identity = re.search(family, model_meta.get("general.name", "").lower())
    companion_identity = re.search(family, candidate_meta.get("general.name", "").lower())
    if main_identity and companion_identity and normalized(main_identity[0]) != normalized(companion_identity[0]):
        return False
    names = [normalized(model.name), normalized(model_meta.get("general.name", ""))]
    other = [normalized(candidate.name), normalized(candidate_meta.get("general.name", ""))]
    if any(name and name not in ("hf", "model", "patchedckpt") and name in other for name in names):
        return True
    if normalized(model.name) == normalized(candidate.name):
        return True
    if candidate_meta.get("general.architecture") == "gemma4-assistant":
        target = re.search(r"gemma[-_ ]?4[-_ ](e?\d+b)", model.name.lower())
        draft = re.search(r"gemma[-_ ]?4[-_ ](e?\d+b)", candidate.name.lower())
        return bool(target and draft and target[1] == draft[1])
    return False


def matching_companions(model, kind):
    model = Path(model)
    main = metadata(str(model))
    arch = main.get("general.architecture", "")
    capability = "vision" if kind == "mmproj" else "mtp"
    if not arch or main.get("aag.capabilities." + capability) is False:
        return []
    matches = []
    for candidate in sorted(model.parent.iterdir()):
        if any(ord(c) < 32 for c in str(candidate)):
            continue
        if candidate.suffix.lower() != ".gguf" or sidecar_kind(candidate) != kind or not candidate.is_file():
            continue
        companion = metadata(str(candidate))
        if not identity_matches(model, candidate, main, companion):
            continue
        if kind == "mmproj":
            if companion.get("general.architecture") != "clip" or companion.get("clip.has_vision_encoder") is not True:
                continue
            width = main.get(arch + ".embedding_length")
            projection = companion.get("clip.vision.projection_dim")
            if not width or not projection:
                continue
            if projection != width:
                # Some Pixtral files have incorrect projection metadata; check the output tensor.
                if companion.get("clip.projector_type") != "pixtral" or tensor_rows(candidate, "mm.2.weight") != width:
                    continue
        else:
            draft_arch = companion.get("general.architecture", "")
            if draft_arch != arch and (arch, draft_arch) != ("gemma4", "gemma4-assistant"):
                continue
            if not companion.get(draft_arch + ".nextn_predict_layers", 0):
                continue
            width = main.get(arch + ".embedding_length")
            draft_width = companion.get(draft_arch + ".embedding_length_out", companion.get(draft_arch + ".embedding_length"))
            if not width or width != draft_width:
                continue
            if not main.get("tokenizer.vocab_size") or main["tokenizer.vocab_size"] != companion.get("tokenizer.vocab_size"):
                continue
        matches.append(str(candidate))
    return matches


def capabilities(model):
    meta = metadata(str(model))
    template = meta.get("tokenizer.chat_template", "")
    mmproj = matching_companions(model, "mmproj")
    mtp = matching_companions(model, "mtp")
    tools = bool(re.search(r"\btools\b", template) and re.search(r"tool_calls|<tool_call>|\[TOOL_CALLS\]|<\|tool_call", template))
    reasoning = bool(("<think>" in template and "</think>" in template) or ("enable_thinking" in template and "<|channel>" in template and ("thought" in template or "<|think|>" in template)) or ("<|channel|>" in template and "analysis" in template))
    vision_arch = meta.get("general.architecture") in {"gemma4", "gemma3n", "qwen2vl", "qwen3vl", "qwen35", "qwen35moe", "mistral3"}
    vision = bool(mmproj or (vision_arch and re.search(r"['\"]image['\"]", template) and re.search(r"image_pad|<\|image\|>|<image>|<start_of_image>", template)))
    values = {"vision": vision, "tools": tools, "reasoning": reasoning}
    for name in values:
        explicit = meta.get("aag.capabilities." + name)
        if isinstance(explicit, bool):
            values[name] = explicit
    values.update(mtp=bool(mtp), mmproj_candidates=mmproj, mtp_candidates=mtp)
    values["badge"] = "".join("[" + (letter if values[key] else " ") + "]" for key, letter in [("vision", "V"), ("tools", "T"), ("reasoning", "R"), ("mtp", "M")])
    return values


def launcher_settings(model):
    meta = metadata(str(model))
    caps = capabilities(model)
    arch = meta.get("general.architecture", "")
    template = re.sub(r"\{#.*?#\}", "", meta.get("tokenizer.chat_template", ""), flags=re.S)
    control = "none"
    if caps["reasoning"]:
        control = "inherent"
        if arch == "gpt-oss" and re.search(r"\{\{[^}]*\breasoning_effort\b", template):
            control = "effort"
        elif re.search(r"\{%[-\s]*(?:if|elif)\b[^%]*\benable_thinking\b", template) and not re.search(r"\bset\s+enable_thinking\s*=\s*(?:true|false)\b", template):
            control = "toggle"
        elif arch == "granite" and re.search(r"\{%[-\s]*(?:if|elif)\s+thinking\b", template):
            control = "thinking"
    maximum = meta.get(arch + ".context_length", 0)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        maximum = 0
    # Keep interactive capacities within the locally exercised 32K envelope.
    maximum = min(maximum, 32768)
    fast_ubatch = 512 if arch == "gemma4" and meta.get("gemma4.embedding_length") == 3840 else 128
    return maximum, control, fast_ubatch, int(caps["vision"]), int(caps["mtp"])


def discover(directory):
    root = Path(directory).resolve()
    for parent, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith("."))
        for name in sorted(files):
            path = Path(parent) / name
            if path.suffix.lower() != ".gguf" or sidecar_kind(path) or any(ord(c) < 32 for c in str(path)):
                continue
            if not path.resolve().is_relative_to(root):
                continue
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name, re.I)
            if shard and shard[1] != "00001":
                continue
            meta = metadata(str(path))
            arch = meta.get("general.architecture", "")
            if arch and arch != "clip" and not arch.endswith("-assistant"):
                yield str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["badges", "inspect", "mmproj", "mtp", "settings", "discover", "mtp3-compatible"])
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    for model in args.models:
        if args.action == "discover":
            for path in discover(model):
                sys.stdout.buffer.write(path.encode() + b"\0")
        elif args.action == "mtp3-compatible":
            meta = metadata(model)
            compatible = meta.get("general.architecture") == "gemma4" and meta.get("gemma4.embedding_length") == 3840 and meta.get("general.file_type") == 2
            print("yes" if compatible else "no")
        elif args.action == "settings":
            print("\t".join(map(str, launcher_settings(model))))
        elif args.action in ("mmproj", "mtp"):
            for path in matching_companions(model, args.action):
                sys.stdout.buffer.write(path.encode() + b"\0")
        else:
            result = capabilities(model)
            print(result["badge"] if args.action == "badges" else json.dumps({"model": model, **result}))


if __name__ == "__main__":
    main()
