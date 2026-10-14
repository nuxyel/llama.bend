# SmolLM2 tokenizer in Bend: pre-tokenizer (SmolLM2's Digits split, then the GPT-2 regex, over bytes) +
# the 48,900 merge rules, applied with the bend-ml-bpe-tokenizer package (proved roundtrip).
#
# Adapted from bend-ml's demos/gpt2/tok.bend (MIT). What changes for SmolLM2: every \p{N} character is its
# own piece; the 21 bytes without a token are dropped before the merges; merge k is id 252 + k; 0x1C-0x1F
# are not whitespace. reference/test_tok.py checks it against `tokenizers`.
#
# The pre-tokenizer decodes UTF-8 and classifies every code point like the regex does
# (\s, \p{L}, \p{N}); the classes of code points >= 128 come from unicode.bend. Known limits: code points
# above U+1FFFF count as letters, and so do CJK ideographs and Hangul (they are letters in Unicode, so this
# is exact); invalid UTF-8 bytes count as letters; a special token written as text is not recognized.

import Base
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE
import ./unicode.bend as U

# ---- reading files ----

def rb_done(m: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(List<&2, U32>):
  (f, r) = m
  do IO<List<&2, U32>>:
    bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    File.close(f)
    return bs

def read_bytes(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_bytes(f, 400000000)
    rb_done(m)

def id_tok2(small: Bool, +n: Nat) -> BPE.Tk:
  match small:
    case True{}:
      BPE.B{n}
    case False{}:
      BPE.M{Nat.sub(n, 256n)}

# numeric id in the bpe/ scheme (byte = 0..255, merge k = 256+k) -> token

def id_tok(+n: Nat) -> BPE.Tk:
  id_tok2(Nat.is_lt(n, 256n), n)

# ---- integers in text: any byte that is not a digit separates ----

def digit2(lo: Bool, hi: Bool, +code: U32) -> Maybe<&2, Nat>:
  match lo hi:
    case True{} True{}:
      Some{U32.to_nat(U32.sub(code, 48))}
    case _ _:
      None{}

def digit(+code: U32) -> Maybe<&2, Nat>:
  digit2(U32.is_le(48, code), U32.is_le(code, 57), code)

def head_digit(xs: List<&2, U32>) -> Maybe<&2, Nat>:
  match xs:
    case Nil{}:
      None{}
    case Con{+c, t}:
      digit(c)

def flush(have: Bool, +cur: Nat, acc: List<&2, Nat>) -> List<&2, Nat>:
  match have:
    case True{}:
      cur <> acc
    case False{}:
      acc

def nums.go(xs: List<&2, U32>, d: Maybe<&2, Nat>, +cur: Nat, have: Bool, acc: List<&2, Nat>) -> List<&2, Nat>:
  match xs d:
    case Nil{} _:
      List.reverse(&2, Nat, flush(have, cur, acc))
    case Con{c, +t} Some{+v}:
      nums.go(t, head_digit(t), Nat.add(Nat.mul(cur, 10n), v), True{}, acc)
    case Con{c, +t} None{}:
      nums.go(t, head_digit(t), 0n, False{}, flush(have, cur, acc))

def nums(+xs: List<&2, U32>) -> List<&2, Nat>:
  nums.go(xs, head_digit(xs), 0n, False{}, Nil{})

# pairs of ids become rules with sequential ids

def rules.go(ns: List<&2, Nat>, +k: Nat) -> List<&2, BPE.Rule>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          Nil{}
        case Con{+b, u}:
          BPE.Rule{k, id_tok(a), id_tok(b)} <> rules.go(u, 1n+k)

# ---- runtime check of the merges table read from a file ----
# rules.go gives the rules the sequential ids 0, 1, 2..., so BPE.wf (distinct ids) holds
# by construction; what a malformed file can still break is the shape of the table.
# It must have an even number of ids, and rule k may only refer to ids that exist
# before it (bytes < 256, or merges 256..256+k-1). The check is linear in the table size.
def ids_ok(ns: List<&2, Nat>, +k: Nat) -> Bool:
  match ns:
    case Nil{}:
      True{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          False{}
        case Con{+b, u}:
          Bool.and(Bool.and(Nat.is_lt(a, Nat.add(256n, k)), Nat.is_lt(b, Nat.add(256n, k))), ids_ok(u, 1n+k))

def toks_of_ids(ns: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      id_tok(n) <> toks_of_ids(t)


# ---- table and permutation of the SmolLM2 base ids ----

def nth(xs: List<&2, Nat>, n: Nat) -> Nat:
  match xs n:
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      nth(t, p)
    case Nil{} _:
      0n

# SmolLM2 id of a token: bytes go through perm (ids 17-251); merge k is 252 + k (17 special tokens and
# 235 byte tokens come first; reference/prep.py checks this for all 48,900 merges)

def gid(+perm: List<&2, Nat>, +t: BPE.Tk) -> Nat:
  match t:
    case BPE.B{+n}:
      nth(perm, n)
    case BPE.M{+k}:
      Nat.add(252n, k)

def gids(+perm: List<&2, Nat>, xs: List<&2, BPE.Tk>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      gid(perm, h) <> gids(perm, t)

# ---- pre-tokenizer: the GPT-2 regex over UTF-8 ----
#   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
# The text is first TAGGED: every byte b becomes b + 256*class + 2048*continuation, where
# the class is the one of its code point (0 letter, 1 digit, 2 other, 3 whitespace; 4 means
# end of list) and continuation marks the 2nd, 3rd and 4th byte of a multibyte character.
# The pre-tokenizer then works on the classes; the tags are removed before the BPE.

def sel(b: Bool, x: Nat, y: Nat) -> Nat:
  match b:
    case True{}:
      x
    case False{}:
      y

def in_range(+c: U32, +lo: U32, +hi: U32) -> Bool:
  Bool.and(U32.is_le(lo, c), U32.is_le(c, hi))

# class of an ASCII byte. Whitespace is the space and 9-13 (\t \n \v \f \r): `tokenizers` does not count
# 0x1C-0x1F as \s (Unicode White_Space does not include them), so they are "other", unlike bend-ml's GPT-2 copy
def class_ascii(+c: U32) -> Nat:
  sel(Bool.or(in_range(c, 65, 90), in_range(c, 97, 122)), 0n, sel(in_range(c, 48, 57), 1n, sel(Bool.or(U32.is_eq(c, 32), in_range(c, 9, 13)), 3n, 2n)))

# class of a code point >= 128: the first range of the sorted table that does not end before it
def after_h(+cp: U32, t: List<&2, U32>) -> Bool:
  match t:
    case Con{lo, Con{+hi, rest}}:
      U32.is_lt(hi, cp)
    case _:
      False{}

def inside_h(+cp: U32, t: List<&2, U32>) -> Bool:
  match t:
    case Con{+lo, Con{+hi, rest}}:
      in_range(cp, lo, hi)
    case _:
      False{}

def cls_go(+cp: U32, tbl: List<&2, U32>, aft: Bool, ins: Bool) -> Nat:
  match tbl aft ins:
    case Con{lo, Con{hi, Con{c, +rest}}} True{} _:
      cls_go(cp, rest, after_h(cp, rest), inside_h(cp, rest))
    case Con{lo, Con{hi, Con{+c, rest}}} False{} True{}:
      U32.to_nat(c)
    case _ _ _:
      0n

def cls_cp(+tbl: List<&2, U32>, +cp: U32) -> Nat:
  cls_go(cp, tbl, after_h(cp, tbl), inside_h(cp, tbl))

def class_cp(+tbl: List<&2, U32>, +cp: U32) -> Nat:
  sel(U32.is_lt(cp, 128), class_ascii(cp), cls_cp(tbl, cp))

# length of the UTF-8 character that starts with byte c (1 for stray or invalid bytes)
def clen(+c: U32) -> Nat:
  sel(U32.is_lt(c, 192), 1n, sel(U32.is_lt(c, 224), 2n, sel(U32.is_lt(c, 240), 3n, sel(U32.is_lt(c, 248), 4n, 1n))))

def byte_at(xs: List<&2, U32>, i: Nat) -> U32:
  match xs i:
    case Con{h, t} 0n:
      h
    case Con{h, t} 1n+p:
      byte_at(t, p)
    case Nil{} _:
      0

def cont_bits(+x: U32) -> U32:
  U32.mod(x, 64)

# the code point of the character of n bytes at the start of xs
def cp_of(+n: Nat, +xs: List<&2, U32>) -> U32:
  match n:
    case 2n:
      U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 32), 64), cont_bits(byte_at(xs, 1n)))
    case 3n:
      U32.add(U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 16), 4096), U32.mul(cont_bits(byte_at(xs, 1n)), 64)), cont_bits(byte_at(xs, 2n)))
    case 4n:
      U32.add(U32.add(U32.add(U32.mul(U32.mod(byte_at(xs, 0n), 8), 262144), U32.mul(cont_bits(byte_at(xs, 1n)), 4096)), U32.mul(cont_bits(byte_at(xs, 2n)), 64)), cont_bits(byte_at(xs, 3n)))
    case _:
      byte_at(xs, 0n)

def tagged(+b: U32, +cls: Nat, +cont: Bool) -> U32:
  U32.add(U32.add(b, U32.mul(U32.from_nat(cls), 256)), U32.mul(U32.from_nat(sel(cont, 1n, 0n)), 2048))

# tags the first n bytes of xs (the 1st is not a continuation), then continues with `rest`
def emit(n: Nat, +cls: Nat, +cont: Bool, xs: List<&2, U32>, rest: List<&2, U32>) -> List<&2, U32>:
  match n xs:
    case 1n+p Con{b, t}:
      tagged(b, cls, cont) <> emit(p, cls, True{}, t, rest)
    case _ _:
      rest

def n_of(+xs: List<&2, U32>) -> Nat:
  clen(byte_at(xs, 0n))

def cls_head(+tbl: List<&2, U32>, +xs: List<&2, U32>) -> Nat:
  class_cp(tbl, cp_of(n_of(xs), xs))

# fuel >= number of bytes left; n and cls describe the character at the start of xs
def tag(fuel: Nat, +tbl: List<&2, U32>, +xs: List<&2, U32>, +n: Nat, +cls: Nat) -> List<&2, U32>:
  match fuel xs:
    case 1n+p Con{b, t}:
      emit(n, cls, False{}, xs, tag(p, tbl, List.drop(&2, U32, xs, n), n_of(List.drop(&2, U32, xs, n)), cls_head(tbl, List.drop(&2, U32, xs, n))))
    case _ _:
      Nil{}

def tag_all(+tbl: List<&2, U32>, +xs: List<&2, U32>) -> List<&2, U32>:
  tag(List.length(&2, U32, xs), tbl, xs, n_of(xs), cls_head(tbl, xs))

def class_of(+c: U32) -> Nat:
  U32.to_nat(U32.mod(U32.div(c, 256), 8))

def head_class(xs: List<&2, U32>) -> Nat:
  match xs:
    case Nil{}:
      4n
    case Con{+c, t}:
      class_of(c)

def head_u32(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{c, t}:
      U32.mod(c, 256)

def tail_l(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{c, t}:
      t

# length (2 or 3) of the contraction at the start of xs, or 0

def contr2(+q: Bool, +b: U32, +c: U32) -> Nat:
  sel(Bool.and(q, Bool.or(Bool.or(U32.is_eq(b, 115), U32.is_eq(b, 116)), Bool.or(U32.is_eq(b, 109), U32.is_eq(b, 100)))), 2n, sel(Bool.and(q, Bool.or(Bool.or(Bool.and(U32.is_eq(b, 114), U32.is_eq(c, 101)), Bool.and(U32.is_eq(b, 118), U32.is_eq(c, 101))), Bool.and(U32.is_eq(b, 108), U32.is_eq(c, 108)))), 3n, 0n))

def contr_len(+xs: List<&2, U32>) -> Nat:
  contr2(U32.is_eq(head_u32(xs), 39), head_u32(tail_l(xs)), head_u32(tail_l(tail_l(xs))))

# splits off the longest prefix of class `cls`: (prefix, rest)

def span(xs: List<&2, U32>, ok: Bool, +cls: Nat, acc: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs ok:
    case Con{c, +t} True{}:
      span(t, Nat.is_eq(head_class(t), cls), cls, c <> acc)
    case _ _:
      (List.reverse(&2, U32, acc), xs)

def span_cls(+xs: List<&2, U32>, +cls: Nat) -> List<&2, U32> & List<&2, U32>:
  span(xs, Nat.is_eq(head_class(xs), cls), cls, Nil{})

def prepend(c: U32, p: List<&2, U32> & List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match p:
    case Tuple{a, b}:
      (c <> a, b)

# whitespace: \s+(?!\S) leaves the last space for the next token if something follows

# number of bytes of the last character of xs (1 + trailing continuation bytes)
def lcl(xs: List<&2, U32>, cur: Nat) -> Nat:
  match xs:
    case Nil{}:
      cur
    case Con{+h, t}:
      lcl(t, sel(U32.is_eq(U32.div(h, 2048), 1), Nat.add(1n, cur), 1n))

def ws4(long: Bool, +run: List<&2, U32>, +rest: List<&2, U32>, +n: Nat, +m: Nat) -> List<&2, U32> & List<&2, U32>:
  match long:
    case False{}:
      (run, rest)
    case True{}:
      (List.take(&2, U32, run, Nat.sub(n, m)), List.append(&2, U32, List.drop(&2, U32, run, Nat.sub(n, m)), rest))

def ws3(+run: List<&2, U32>, +rest: List<&2, U32>, ends: Bool, +n: Nat) -> List<&2, U32> & List<&2, U32>:
  match ends:
    case True{}:
      (run, rest)
    case False{}:
      ws4(Nat.is_lt(lcl(run, 0n), n), run, rest, n, lcl(run, 0n))

# a run that ends the text, or ends right before a digit, is taken whole: SmolLM2 splits every digit off
# first, so the regex sees the run at the end of its segment and \s+(?!\S) matches all of it
def ws2(p: List<&2, U32> & List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match p:
    case Tuple{+run, +rest}:
      ws3(run, rest, Bool.or(Nat.is_eq(head_class(rest), 4n), Nat.is_eq(head_class(rest), 1n)), List.length(&2, U32, run))

def ws(xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  ws2(span_cls(xs, 3n))

# a digit (class 1) is one character: the byte and its continuation bytes (tag bit 2048)
def is_cont(xs: List<&2, U32>) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+c, t}:
      U32.is_eq(U32.div(c, 2048), 1)

def span_cont(xs: List<&2, U32>, ok: Bool, acc: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs ok:
    case Con{c, +t} True{}:
      span_cont(t, is_cont(t), c <> acc)
    case _ _:
      (List.reverse(&2, U32, acc), xs)

def one_char(xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs:
    case Nil{}:
      (Nil{}, Nil{})
    case Con{c, +t}:
      prepend(c, span_cont(t, is_cont(t), Nil{}))

# 0 contraction, 1 run of its own class, 2 space + run, 3 run of whitespace, 4 one digit.
# SmolLM2 splits every \p{N} character off before the GPT-2 regex (Digits(individual_digits)), so a digit is
# always a piece of its own and a space is never attached to it (no " ?\p{N}+").

def kind_of(n_contr: Nat, +cc: Nat, is_sp: Bool, +hc: Nat) -> Nat:
  sel(Nat.is_lt(0n, n_contr), 0n, sel(Nat.is_eq(cc, 1n), 4n, sel(Bool.not(Nat.is_eq(cc, 3n)), 1n, sel(Bool.and(is_sp, Bool.and(Nat.is_lt(hc, 3n), Bool.not(Nat.is_eq(hc, 1n)))), 2n, 3n))))

def nt(kind: Nat, +xs: List<&2, U32>, +n_contr: Nat) -> List<&2, U32> & List<&2, U32>:
  match kind:
    case 0n:
      (List.take(&2, U32, xs, n_contr), List.drop(&2, U32, xs, n_contr))
    case 1n:
      span_cls(xs, head_class(xs))
    case 2n:
      prepend(head_u32(xs), span_cls(tail_l(xs), head_class(tail_l(xs))))
    case 3n:
      ws(xs)
    case _:
      one_char(xs)

def next_token(+xs: List<&2, U32>) -> List<&2, U32> & List<&2, U32>:
  match xs:
    case Nil{}:
      (Nil{}, Nil{})
    case Con{+c, t}:
      nt(kind_of(contr_len(xs), class_of(c), U32.is_eq(U32.mod(c, 256), 32), head_class(t)), xs, contr_len(xs))

# splits the text into words; `fuel` >= number of tokens; pr = next_token(xs)

def split(fuel: Nat, xs: List<&2, U32>, pr: List<&2, U32> & List<&2, U32>) -> List<&2, List<&2, U32>>:
  match fuel pr:
    case 0n _:
      Nil{}
    case 1n+p Tuple{tok, +rest}:
      match tok:
        case Nil{}:
          Nil{}
        case Con{h, t}:
          Con{h, t} <> split(p, rest, next_token(rest))

def pretok(+xs: List<&2, U32>) -> List<&2, List<&2, U32>>:
  split(List.length(&2, U32, xs), xs, next_token(xs))

# ---- encoding the whole text ----

def nats_of(xs: List<&2, U32>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      U32.to_nat(U32.mod(h, 256)) <> nats_of(t)

# 21 bytes have no token in SmolLM2 (perm gives 0 for them): `tokenizers` drops them from a piece before the
# merges (no unk token, no byte fallback), so they are dropped here too
def kb_sel(keep: Bool, +n: Nat, rest: List<&2, Nat>) -> List<&2, Nat>:
  match keep:
    case True{}:
      n <> rest
    case False{}:
      rest

def known_bytes(+perm: List<&2, Nat>, ns: List<&2, Nat>) -> List<&2, Nat>:
  match ns:
    case Nil{}:
      Nil{}
    case Con{+n, t}:
      kb_sel(Bool.not(Nat.is_eq(nth(perm, n), 0n)), n, known_bytes(perm, t))

def encode_words(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ws: List<&2, List<&2, U32>>) -> List<&2, Nat>:
  match ws:
    case Nil{}:
      Nil{}
    case Con{w, t}:
      List.append(&2, Nat, gids(perm, BPE.encode(table, BPE.lift(known_bytes(perm, nats_of(w))))), encode_words(table, perm, t))

def encode_text(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +bs: List<&2, U32>) -> List<&2, Nat>:
  encode_words(table, perm, pretok(tag_all(U.non_letters(), bs)))

# ---- decoding: SmolLM2 ids back to UTF-8 text ----
# ids 0-16 are special tokens: they are left out, as `tokenizers`' decode(skip_special_tokens = True) does;
# ids 17-251 are bytes (the byte whose perm entry is the id); id 252 + k is merge k. Bytes that do not form
# valid UTF-8 become U+FFFD, the replacement character Python's decode(errors = "replace") gives.

def index_of(perm: List<&2, Nat>, +id: Nat, +i: Nat) -> Nat:
  match perm:
    case Nil{}:
      0n
    case Con{+p, t}:
      sel(Nat.is_eq(p, id), i, index_of(t, id, 1n+i))

def tk_sel(b: Bool, x: BPE.Tk, y: BPE.Tk) -> BPE.Tk:
  match b:
    case True{}:
      x
    case False{}:
      y

def tk_of_gid(+perm: List<&2, Nat>, +id: Nat) -> BPE.Tk:
  tk_sel(Nat.is_lt(id, 252n), BPE.B{index_of(perm, id, 0n)}, BPE.M{Nat.sub(id, 252n)})

def tks_sel(special: Bool, +perm: List<&2, Nat>, +id: Nat, rest: List<&2, BPE.Tk>) -> List<&2, BPE.Tk>:
  match special:
    case True{}:
      rest
    case False{}:
      tk_of_gid(perm, id) <> rest

def tks_of(+perm: List<&2, Nat>, ids: List<&2, Nat>) -> List<&2, BPE.Tk>:
  match ids:
    case Nil{}:
      Nil{}
    case Con{+h, t}:
      tks_sel(Nat.is_lt(h, 17n), perm, h, tks_of(perm, t))

def u32s(xs: List<&2, Nat>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{n, t}:
      U32.from_nat(n) <> u32s(t)

def head_b(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{c, t}:
      c

def tail1(xs: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{c, t}:
      t

def selu(b: Bool, x: U32, y: U32) -> U32:
  match b:
    case True{}:
      x
    case False{}:
      y

def is_cont_b(+b: U32) -> Bool:
  Bool.and(U32.is_le(128, b), U32.is_le(b, 191))

# the length of a UTF-8 sequence from its first byte; 1 for a byte that cannot start one (a continuation byte,
# 0xC0, 0xC1, 0xF5-0xFF)
def ulen(+b: U32) -> Nat:
  sel(U32.is_lt(b, 194), 1n, sel(U32.is_lt(b, 224), 2n, sel(U32.is_lt(b, 240), 3n, sel(U32.is_lt(b, 245), 4n, 1n))))

# the range of the second byte after a first byte b0 (Unicode table 3-7: no overlong forms, no surrogates,
# nothing above U+10FFFF); the bytes after it are always 0x80-0xBF
def lo2(+b0: U32) -> U32:
  selu(U32.is_eq(b0, 224), 160, selu(U32.is_eq(b0, 240), 144, 128))

def hi2(+b0: U32) -> U32:
  selu(U32.is_eq(b0, 237), 159, selu(U32.is_eq(b0, 244), 143, 191))

# how many of the (at most n) bytes after the first are valid continuation bytes, in a row
def conts(n: Nat, t: List<&2, U32>) -> Nat:
  match n t:
    case 1n+p Con{+c, r}:
      sel(is_cont_b(c), 1n+conts(p, r), 0n)
    case _ _:
      0n

def conts2(n: Nat, +lo: U32, +hi: U32, t: List<&2, U32>) -> Nat:
  match n t:
    case 1n+p Con{+c, r}:
      sel(in_range(c, lo, hi), 1n+conts(p, r), 0n)
    case _ _:
      0n

def dec_cp(+b0: U32, +b1: U32, +b2: U32, +b3: U32, n: Nat) -> U32:
  match n:
    case 1n:
      selu(U32.is_lt(b0, 128), b0, 65533)
    case 2n:
      (U32.sub(b0, 192) * 64 + U32.sub(b1, 128) : U32)
    case 3n:
      (U32.sub(b0, 224) * 4096 + U32.sub(b1, 128) * 64 + U32.sub(b2, 128) : U32)
    case _:
      (U32.sub(b0, 240) * 262144 + U32.sub(b1, 128) * 4096 + U32.sub(b2, 128) * 64 + U32.sub(b3, 128) : U32)

# the next character and the bytes after it
type Next is Data:
  NDone{}
  NChar{c: Char, rest: List<&2, U32>}

# a complete sequence of n bytes is one character; an incomplete one (k < n - 1 continuation bytes) is one
# U+FFFD for its first 1 + k bytes, as Rust's String::from_utf8_lossy (the decoder of `tokenizers`) does
def nchar(ok: Bool, +b0: U32, +t: List<&2, U32>, +n: Nat, +k: Nat) -> Next:
  match ok:
    case True{}:
      NChar{Chr{dec_cp(b0, head_b(t), head_b(tail1(t)), head_b(tail1(tail1(t))), n)}, List.drop(&2, U32, t, k)}
    case False{}:
      NChar{Chr{65533}, List.drop(&2, U32, t, k)}

def next_char(bs: List<&2, U32>) -> Next:
  match bs:
    case Nil{}:
      NDone{}
    case Con{+b0, +t}:
      nchar(Nat.is_eq(conts2(Nat.sub(ulen(b0), 1n), lo2(b0), hi2(b0), t), Nat.sub(ulen(b0), 1n)), b0, t, ulen(b0), conts2(Nat.sub(ulen(b0), 1n), lo2(b0), hi2(b0), t))

# fuel: one character per byte at most
def utf8_chars(fuel: Nat, nx: Next) -> List<&2, Char>:
  match fuel nx:
    case 1n+p NChar{c, r}:
      c <> utf8_chars(p, next_char(r))
    case _ _:
      Nil{}

def decode_bytes(+bs: List<&2, U32>) -> String:
  String.from_list(utf8_chars(List.length(&2, U32, bs), next_char(bs)))

def decode_text(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ids: List<&2, Nat>) -> String:
  decode_bytes(u32s(BPE.decode(table, tks_of(perm, ids))))

# ---- test CLI: ids of the text in a file ----

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def go_ok(ok: Bool, +ns: List<&2, Nat>, +pm: List<&2, U32>, tx: List<&2, U32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(show_nats(encode_text(rules.go(ns, 0n), nums(pm), tx)))
    case False{}:
      IO.print("invalid merges table")

def go(+m: List<&2, U32>, +pm: List<&2, U32>, tx: List<&2, U32>) -> IO(Unit):
  go_ok(ids_ok(nums(m), 0n), nums(m), pm, tx)

def run3(a: String, b: String, c: String) -> IO(Unit):
  do IO<Unit>:
    m : List<&2, U32> <- read_bytes(a)
    pm : List<&2, U32> <- read_bytes(b)
    tx : List<&2, U32> <- read_bytes(c)
    go(m, pm, tx)

def dec_ok(ok: Bool, +ns: List<&2, Nat>, +pm: List<&2, U32>, ids: List<&2, U32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print(decode_text(rules.go(ns, 0n), nums(pm), nums(ids)))
    case False{}:
      IO.print("invalid merges table")

def dec_go(+m: List<&2, U32>, +pm: List<&2, U32>, ids: List<&2, U32>) -> IO(Unit):
  dec_ok(ids_ok(nums(m), 0n), nums(m), pm, ids)

# the text of the ids (decimal, separated by spaces) in a file
def run_dec(a: String, b: String, c: String) -> IO(Unit):
  do IO<Unit>:
    m : List<&2, U32> <- read_bytes(a)
    pm : List<&2, U32> <- read_bytes(b)
    ids : List<&2, U32> <- read_bytes(c)
    dec_go(m, pm, ids)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{a, Con{b, Con{c, Nil{}}}}}:
      run3(a, b, c)
    case Con{prog, Con{a, Con{b, Con{d, Con{c, Nil{}}}}}}:
      run_dec(a, b, c)
    case _:
      IO.print("usage: tok.bend merges_num.txt perm.txt text, or tok.bend merges_num.txt perm.txt --decode ids")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
