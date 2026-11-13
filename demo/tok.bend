# The command line of SmolLM2's tokenizer (llama/tok.bend), which the tests drive: reference/test_tok.py compares the
# ids with `tokenizers`, test_detok.py the decoded text, and test_chat.py the chat template with transformers'.
#
#   tok merges_num.txt perm.txt <file>                the ids of the file's text
#   tok merges_num.txt perm.txt --decode <file>       the text of the ids in the file (decimal, separated by spaces)
#   tok merges_num.txt perm.txt --chat <file>         the ids of the conversation in the file, in SmolLM2-Instruct's
#   tok merges_num.txt perm.txt --chat-nogen <file>   chat template, with and without the assistant's turn opened
#
# A malformed merges table is refused (`invalid merges table`).

import Base
import ../llama/tok.bend as K

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

# a file of messages: role and content, each followed by a 0 byte; z: whether the head of bs is a 0 byte
def head_zero(bs: List<&2, U32>) -> Bool:
  match bs:
    case Nil{}:
      False{}
    case Con{+b, t}:
      U32.is_eq(b, 0)

def split0(bs: List<&2, U32>, z: Bool, cur: List<&2, U32>) -> List<&2, List<&2, U32>>:
  match bs z:
    case Nil{} _:
      Nil{}
    case Con{b, +t} True{}:
      List.reverse(&2, U32, cur) <> split0(t, head_zero(t), Nil{})
    case Con{+b, +t} False{}:
      split0(t, head_zero(t), b <> cur)

def messages(+bs: List<&2, U32>) -> List<&2, List<&2, U32>>:
  split0(bs, head_zero(bs), Nil{})

# what the file holds: a text to encode, ids to decode, or a conversation (gen: with the assistant's turn opened)
type Mode is Data:
  Enc{}
  Dec{}
  Chat{gen: Bool}

def out(mode: Mode, tb: K.Tables, bs: List<&2, U32>) -> String:
  match mode tb:
    case Enc{} K.Tables{rules, perm}:
      show_nats(K.encode_text(rules, perm, bs))
    case Dec{} K.Tables{rules, perm}:
      K.decode_text(rules, perm, K.nums(bs))
    case Chat{gen} K.Tables{rules, perm}:
      show_nats(K.chat_ids(rules, perm, messages(bs), gen))

def print_out(mt: Maybe<&2, K.Tables>, mode: Mode, bs: List<&2, U32>) -> IO(Unit):
  match mt:
    case Some{tb}:
      IO.print(out(mode, tb, bs))
    case None{}:
      IO.print("invalid merges table")

# a: merges_num.txt, b: perm.txt, c: the file
def run(a: String, b: String, c: String, mode: Mode) -> IO(Unit):
  do IO<Unit>:
    mt : Maybe<&2, K.Tables> <- K.read_tables(a, b)
    bs : List<&2, U32> <- K.read_bytes(c)
    print_out(mt, mode, bs)

def usage() -> IO(Unit):
  IO.print("usage: tok.bend merges_num.txt perm.txt text, or tok.bend merges_num.txt perm.txt --decode ids, or tok.bend merges_num.txt perm.txt --chat|--chat-nogen messages")

def run_flag(dec: Bool, chat: Bool, nogen: Bool, a: String, b: String, c: String) -> IO(Unit):
  match dec chat nogen:
    case True{} _ _:
      run(a, b, c, Dec{})
    case False{} True{} _:
      run(a, b, c, Chat{True{}})
    case False{} False{} True{}:
      run(a, b, c, Chat{False{}})
    case False{} False{} False{}:
      usage()

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{a, Con{b, Con{c, Nil{}}}}}:
      run(a, b, c, Enc{})
    case Con{prog, Con{a, Con{b, Con{+d, Con{c, Nil{}}}}}}:
      run_flag(String.eq(d, "--decode"), String.eq(d, "--chat"), String.eq(d, "--chat-nogen"), a, b, c)
    case _:
      usage()

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
