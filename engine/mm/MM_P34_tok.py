#!/usr/bin/env python3
"""MM P34 — standalone GGUF tokenizer (parse_gguf_kv + SimpleTokenizer
extracted VERBATIM from engine0/api_server.py; no fastapi dependency)."""
import re, struct, sys, unicodedata
import itertools as _it

def parse_gguf_kv(path):
  f = open(path, "rb")
  assert f.read(4) == b"GGUF", "not a gguf"
  struct.unpack("<I", f.read(4))
  struct.unpack("<Q", f.read(8))
  n_kv, = struct.unpack("<Q", f.read(8))
  def rd_str():
    n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
  def rd_val(t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return bool(f.read(1)[0])
    if t == 8: return rd_str()
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    if t == 9:
      et, = struct.unpack("<I", f.read(4)); cnt, = struct.unpack("<Q", f.read(8))
      return [rd_val(et) for _ in range(cnt)]
    raise ValueError(f"gguf val type {t}")
  kv = {}
  for _ in range(n_kv):
    k = rd_str(); t, = struct.unpack("<I", f.read(4)); kv[k] = rd_val(t)
  f.close()
  return kv

class SimpleTokenizer:
  def __init__(self, normal_tokens: dict, special_tokens: dict, preset: str = "llama3",
               bos_id=None, eos_id: int = 0, eot_id=None):
    preset = {"qwen35": "qwen2", "qwen35moe": "qwen2"}.get(preset, preset)
    if preset not in ("llama3", "llama-v3", "llama-bpe", "qwen2", "olmo", "kimi-k2", "tekken", "glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i, b in enumerate(b for b in range(256) if b not in bs)}
    def ucat_range(pre: str) -> str:
      cps = enumerate(cp for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
      runs = [list(g) for _, g in _it.groupby(cps, lambda e: e[1] - e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv: dict):
    normal, special = [], []
    for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]):
      (normal if kv["tokenizer.ggml.token_type"][idx] == 1 else special).append((tok, idx))
    special_dict = dict(special)
    return SimpleTokenizer(dict(normal), special_dict, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_dict.get('<|im_end|>')))

  def _encode_word(self, word: bytes):
    if (early_token := self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")

  def _encode_sentence(self, chunk: str):
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]

  def encode(self, text: str):
    tokens, pos = [], 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids): return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def is_end(self, token_id: int): return token_id in (self.eos_id, self.eot_id)
