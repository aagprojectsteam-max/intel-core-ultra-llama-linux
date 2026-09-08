"""Bounded GGUF v2/v3 header reader; tensor payloads are never read."""

import math
import os
import struct


class Header:
    def __init__(self, stream):
        self.stream = stream
        self.limit = min(os.fstat(stream.fileno()).st_size, 256 * 1024 * 1024)

    def read(self, size):
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError("GGUF header exceeds file or safety limit")
        value = self.stream.read(size)
        if len(value) != size:
            raise ValueError("Truncated GGUF header")
        return value

    def skip(self, size):
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError("GGUF header exceeds file or safety limit")
        self.stream.seek(size, 1)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep=True):
        length = self.number("Q")
        if length > 16 * 1024 * 1024:
            raise ValueError("GGUF string exceeds safety limit")
        if keep:
            return self.read(length).decode("utf-8")
        self.skip(length)

    def value(self, kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            return self.number(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if count > 2_000_000:
                raise ValueError("GGUF array exceeds safety limit")
            if element == 8:
                for _ in range(count):
                    self.string(keep=False)
            elif element in formats:
                self.skip(count * struct.calcsize("<" + formats[element]))
            else:
                raise ValueError("Unsupported GGUF array type")
            return count
        raise ValueError("Unsupported GGUF value type")


def read_header(path, tensor_name=None):
    with open(path, "rb") as stream:
        header = Header(stream)
        if header.read(4) != b"GGUF" or header.number("I") not in (2, 3):
            raise ValueError("Expected GGUF v2 or v3")
        tensors, entries = header.number("Q"), header.number("Q")
        if tensors > 100_000 or entries > 100_000:
            raise ValueError("GGUF entry count exceeds safety limit")
        values = {}
        for _ in range(entries):
            key, kind = header.string(), header.number("I")
            value = header.value(kind)
            if key in values:
                raise ValueError("Duplicate GGUF metadata key")
            if kind != 9:
                values[key] = value
            elif key == "tokenizer.ggml.tokens":
                values["tokenizer.vocab_size"] = value
        if tensor_name is None:
            return values
        for _ in range(tensors):
            name, dimensions = header.string(), header.number("I")
            if dimensions < 1 or dimensions > 4:
                raise ValueError("Unsupported tensor dimensions")
            shape = [header.number("Q") for _ in range(dimensions)]
            header.read(12)
            if name == tensor_name:
                return math.prod(shape[1:])
        return None
