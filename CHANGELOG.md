# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions have four
parts, `MAJOR.MINOR.PATCH.BUILD`, as in the SysManage projects.

## [Unreleased]

### Added
- M0 command-line contract: `proselign --voice NAME` filter plus `train`,
  `voices`, `inspect`, `export`, `import` and `delete`.
- Stylometric voice profiles (stdlib only) stored privately under
  `$XDG_DATA_HOME/proselign`.
- Markdown-aware segmentation; protected-token fidelity checks with a bounded
  repair loop and `--strict`.
- Backends: OpenAI-compatible local endpoint (llama.cpp, vLLM, Ollama) and a
  deterministic mock.
- MITRE "Lucky 13" security test suite.
- Ubuntu packaging for 22.04, 24.04 and 26.04 LTS (amd64, arm64) and the
  `ppa:bceverly/proselign` release pipeline.
- Project website (proselign.org).
- `train` reads Word (`.docx`, and `.doc` from Word 6 through 2003) and PDF
  documents, extracting their text in pure Python; unreadable files are
  skipped with a warning.
- `train` reads `.mbox` mailboxes of sent mail: each message is a document,
  and quoted replies, forwarded messages, signatures, links and attachments
  are left out.

- `~/.proselignrc` settings file (sample in `examples/proselignrc`): default
  voice, strength, format and backend, plus style switches for passive voice,
  en dashes, em dashes, exclamation marks, semicolons and an AI-artifact
  filter. `--config FILE`, `--no-config` and `PROSELIGN_CONFIG` select it.
- Rewrite Word documents in place: `proselign --voice NAME in.docx -o
  out.docx` rewrites body and footnote prose while styles, headers, footers,
  tables, images and run formatting (bold, italic, ...) are preserved
  byte for byte. `--output`/`-o` also writes text results to a file.
- `first_person = plural` in `~/.proselignrc` makes every rewrite speak as
  "we/us/our" instead of "I/me/my" (`singular` and `voice` are the other
  choices); stray singular pronouns are sent back for another attempt.
- Markdown rewrites keep headings exactly as written, and training on `.md`
  files measures only the prose, not the markup.
- Rewrites that drift into another language mid-sentence (a known habit of
  some local models) are caught: the stray fragment is translated back on the
  fly, or the chunk is retried, and never written out.
- A comma or full stop that the model moves across a closing quote mark is
  put back, so faithful quotations are no longer rejected for it. A full
  stop added after a quotation that already ends in one (`RISK.".`) is
  dropped.
- A Word document's title or cover page is left completely untouched: a cover
  page from Word's gallery, or a short first page (under 250 words) ending in
  a page or section break. `keep_title_page = off` in `~/.proselignrc`
  rewrites it too.
- Warnings are summarized in at most two lines, with a count per reason;
  `--verbose` lists them all.
- Voices learn more of the author: vocabulary habits (preferred plain words,
  model-favoured words never used, frequent sentence openers, serial comma,
  spelling) and up to 200 typical passages, which are shown to the model as
  examples chosen by topic (`--examples`, default 3). Passages stay local and
  are left out of exports. Retrain a voice to gain them.
- Rewrites are measured against the voice: one round of feedback when sentence
  length or contractions are far off, and `--candidates N` keeps the closest
  of N rewrites (a function-word distance in the style of Burrows' Delta).
- `proselign eval` round-trips sample passages through neutral prose and back
  and reports how much of the gap to the author's own writing was closed, so
  models and settings can be compared. Settings that depart from the
  author's habits on purpose (`passive_voice = off`, a forced `first_person`)
  are set aside for the measurement, and the report names them. The neutral
  versions are kept with the voice (never exported) and reused by later runs,
  so models are compared from the same starting texts; `--fresh` renews them.
  A control rewrite with no voice (told only to keep each text's own style)
  separates what the voice contributes from the change any rewrite makes
  (`--no-control` skips it). `--repeats N` averages N runs and reports their
  range, and a breakdown table shows what function words, sentence length,
  word length and contractions each contribute to the distance. `--ablate`
  rewrites again without each part of the voice in turn (sentence length,
  contractions, person, passive voice, word length, punctuation, vocabulary,
  tone, the example passages, the steering feedback) and reports whether
  each part helps or hurts.
- A model the endpoint doesn't have (HTTP 404) is reported as such, with the
  `ollama pull` command to fetch it, instead of "cannot reach inference
  endpoint"; other HTTP errors carry the server's own message.
- Replies are capped at about twice the request's length, so a model that
  never stops (Ollama keeps generating past its context window) fails fast
  and the chunk is retried, instead of hanging until the timeout. A timeout
  or dropped connection is reported plainly instead of as a traceback.
- An animated progress line with elapsed time and an estimate of the time
  left, on by default when stderr is a terminal; `--no-progress` or
  `progress = off` in `~/.proselignrc` hides it. When the run succeeds, the
  line is replaced by the total time taken (`proselign: finished in 4:12`).
- `proselign tune --voice NAME` fine-tunes a model on a voice: passages of
  the voice's documents are flattened into neutral prose by the current
  model, a LoRA adapter is trained to turn them back into the author's
  writing (QLoRA on one CUDA GPU; 16 GB fits a 14B model), and the adapter is
  registered with Ollama as `proselign-NAME` on the same base model.
  Rewrites and `eval` in that voice then use it, with a short prompt and no
  profile or examples (`--no-tuned` compares without it); `eval` uses only
  the 20 passages held out of training. Facts are masked on both sides of
  every training pair, so the tuned model learns to keep placeholders. The
  pairs are kept with the voice and reused; `--pairs-only` stops after them.
  Training needs the optional extras: `pip install 'proselign[tune]'`.
- `--parallel N` (and `parallel` in `~/.proselignrc`, default 4) sends that
  many chunks to the model at once; a server that runs requests side by side
  (Ollama with `OLLAMA_NUM_PARALLEL=4`) gets through them several times
  faster.
- `train --min-words N` leaves out documents and e-mail messages shorter
  than N words, so a voice can be learned from considered writing rather
  than quick notes.
- Rewrites are reviewed sentence by sentence. Each changed sentence must keep
  its figures, names, quotations, negations, modal verbs and persons and
  invent no name; then a model (the untuned base model by default;
  `--judge-model` or `judge_model` in `~/.proselignrc` picks another) is
  asked only whether the two say the same thing. A sentence that fails gets
  the author's original back, so one bad sentence no longer costs the good
  edits around it. Dropped sentences are restored, added ones removed, and
  list markers, the case of a sentence's first word and the whitespace
  between sentences always come from the original. `--no-review` (or
  `review = off`) turns this off.
- Only sentences that need work are changed: one that breaks a style rule,
  runs far longer than the author's sentences, or spells out what the author
  contracts. The model is told which ones and why, and every other sentence
  is left exactly as written; a chunk with none is not sent at all.
  `--every-sentence` (or `every_sentence = on`) rewrites everything.

### Changed
- Rewrites are several times faster: chunks are about a paragraph (400
  tokens, `--max-chunk-tokens`) instead of 1,500 tokens, so a failed one
  costs little to redo and every request fits a 4,096-token context; a
  failed chunk is retried once instead of twice; and the extra steering
  round toward the profile runs only at strength 0.75 and above, where it
  helps.
- Rewrites are rejected (and the passage kept as written) when they add,
  drop or swap a negation or modal verb ("expect it" becoming "it could
  be"), drop a sentence or add one the original has nothing like, or add or
  drop an acronym such as MSD, or add a figure the original does not have
  ("five more months" written as "5 more months"). Plain-text headings ("5. Intercompany paper
  and where the IP sits") are never sent to the model, and list labels
  ("1. Period. Your list ...") are masked like facts.
- A chunk that loses a fact is no longer kept whole: it is retried one
  paragraph at a time (a list one item at a time, its markers untouched), so
  only the pieces that still fail are kept as written. Greetings, sign-offs
  and all-caps headings are never sent alone, and e-mail headers (`To:`,
  `Cc:`, `Subject:`) are never rewritten.
- Numbers, document identifiers such as Bates numbers (`INDIV_DEF_00004939`,
  `F5_00213467`), quotations, URLs and e-mail addresses never reach the
  model: each is sent as a placeholder (`{{3}}`) and put back afterwards, so
  the model cannot drop a digit or reword a quotation. A rewrite far longer
  or shorter than its source (invented or dropped content), one that adds
  or drops a name (writing "Korotkoff" where the original said "he"), and
  one that uses I/we/our more often or you/your differently than the source
  (making the author the subject of what someone else did) are rejected.
- `first_person = plural` now governs only how the author refers to
  themselves, and `passive_voice = off` asks for whoever the original says
  acted, never "we"; where the original names no one, the sentence stays
  passive.
- Bold the original did not have, paragraph breaks added inside a single
  paragraph or list (or between list items that follow an introduction), and closing pleasantries ("Hope everything's going
  well") are removed from rewrites.
- `proselign export` writes archives owner-only (0600) instead of 0644.
- Voices reach the model as concrete writing directives (sentence length,
  contractions, person, punctuation) instead of raw metrics, which steers
  rewrites far more reliably; style breaches are fed back for another try.
- The model is told never to add greetings, sign-offs or other content.
- A number followed by a comma ("42, then") is no longer read as "42,".
