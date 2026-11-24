"""Script features w (paper Sec. 3.4 "input"; Supp. C "input"): per-frame word timing features (11) and viseme
ids (13 visemes from grapheme-to-phoneme conversion, plus silence and "other").
"""
import functools
import json
import math
import numpy as np
import os
import re
import unicodedata


HZ = 30


SENTENCE_END = tuple(".?!。？！…")


SILENT = "silent"


def sentences_from_words(words: list, sentence_end: tuple) -> list:
    """[{"w", "start_frame", "end_frame"}] -> [(first start_frame, last end_frame)] per sentence."""
    out, start = [], None
    for w in words:
        if start is None:
            start = int(w["start_frame"])
        if w["w"].endswith(sentence_end):
            out.append((start, int(w["end_frame"])))
            start = None
    if start is not None:
        out.append((start, int(words[-1]["end_frame"])))
    return out


N_TIMING = 11


DIST_CLIP = 30


OTHER = 27


LEXICAL = ("letters", "viseme_v2")  # the paper uses viseme_v2 (viseme ids); letters = an earlier letter embedding


def _anyascii():
    from anyascii import anyascii
    return anyascii


def word_letters(w: str, romanise=None) -> list:
    """Letter ids of one word (module doc)."""
    romanise = romanise or _anyascii()
    out = []
    for ch in romanise(w).lower():
        if "a" <= ch <= "z":
            out.append(ord(ch) - ord("a") + 1)
        elif "0" <= ch <= "9":
            out.append(OTHER)
    return out or [OTHER]


def features(track: dict, n_frames: int, letters: bool = True, lexical: str = "letters") -> tuple:
    """(timing (T, 11) float32, viseme ids (T,) int16) of a script (words with start / end frames, marks,
    language).  Timing columns: in_word, word_onset, word_offset, pos_in_word, log_word_len, pause, sentence_end,
    punct, to_next_onset, since_prev_end, text_present (Supp. C "input").  lexical="viseme_v2" (the paper) returns
    viseme ids; letters=False returns timing only."""
    if lexical not in LEXICAL:
        raise ValueError("lexical must be one of %s" % (LEXICAL,))
    if not letters and lexical == "viseme_v2":
        raise ValueError("letters=False (timing only) takes no lexical='viseme_v2'; call features(track, T, letters=False)")
    if letters and lexical == "viseme_v2":
        return features(track, n_frames, letters=False)[0], viseme_features(track, n_frames)["viseme"]
    x = np.zeros((n_frames, N_TIMING), dtype=np.float32)
    ids = np.zeros(n_frames, dtype=np.int16)
    words = track["words"]
    silent = track.get(SILENT, False)
    if silent is not False and (silent is not True or words):
        raise ValueError("silent must be true / false, and true only without words")
    if not words:
        if silent:
            x[:, 8:11] = 1
        return x, ids
    romanise = _anyascii() if letters else None
    starts = np.array([w["start_frame"] for w in words])
    ends = np.array([w["end_frame"] for w in words])
    if not (np.all(starts <= ends) and np.all(starts[1:] > ends[:-1]) and starts[0] >= 0 and ends[-1] < n_frames):
        raise ValueError("words are not ordered, non-overlapping frames inside [0, %d)" % n_frames)
    marks = track["marks"]
    word_marks = {}
    for m in marks:
        if m["kind"] in ("sentence_end", "punct"):
            word_marks.setdefault(m["kind"], set()).add(m["word"])
    for i, (s, e) in enumerate(zip(starts, ends)):
        n = e - s + 1
        x[s:e + 1, 0] = 1
        x[s, 1] = 1
        x[e, 2] = 1
        x[s:e + 1, 3] = (np.arange(n) / max(n - 1, 1)).astype(np.float32)
        x[s:e + 1, 4] = math.log(n)
        if i in word_marks.get("sentence_end", ()):
            stop = starts[i + 1] if i + 1 < len(words) else n_frames
            x[s:stop, 6] = 1
        if i in word_marks.get("punct", ()):
            x[s:e + 1, 7] = 1
        if letters:
            lt = word_letters(words[i]["w"], romanise)
            ids[s:e + 1] = np.asarray(lt, dtype=np.int16)[(np.arange(n) * len(lt)) // n]
    for m in marks:
        if m["kind"] == "pause":
            x[m["start_frame"]:m["end_frame"] + 1, 5] = 1
    t = np.arange(n_frames)
    k = np.searchsorted(starts, t, side="left")  # first start >= t
    nxt = np.where(k < len(starts), starts[np.minimum(k, len(starts) - 1)] - t, DIST_CLIP)
    j = np.searchsorted(ends, t, side="right") - 1  # last end <= t
    prv = np.where(j >= 0, t - ends[np.maximum(j, 0)], DIST_CLIP)
    x[:, 8] = np.minimum(nxt, DIST_CLIP) / DIST_CLIP
    x[:, 9] = np.minimum(prv, DIST_CLIP) / DIST_CLIP
    x[:, 10] = 1
    return x, ids


VISEMES = ("sil", "PP", "FF", "TH", "DD", "KK", "CH", "SS", "RR", "AA", "EE", "II", "OO", "UU", "XX")


V = {name: i for i, name in enumerate(VISEMES)}


SIL, VISEME_OTHER = V["sil"], V["XX"]


VOWEL_WEIGHT, CONSONANT_WEIGHT = 2, 1


_SCALE = 6  # integer weights: a vowel split over k <= 3 visemes stays integral


G2P_METHODS = ("cmudict", "g2p_en", "pypinyin", "es_rules")


NLTK_DATA = os.environ.get("STRATA_NLTK_DATA", os.path.join(os.path.expanduser("~"), "nltk_data"))  # cmudict + averaged_perceptron_tagger (g2p_en needs both on import)


EN_VOWELS = {"AA": ("AA",), "AE": ("AA",), "AH": ("AA",), "AO": ("OO",), "AW": ("AA", "UU"), "AY": ("AA", "II"),
             "EH": ("EE",), "ER": ("RR",), "EY": ("EE", "II"), "IH": ("II",), "IY": ("II",), "OW": ("OO", "UU"),
             "OY": ("OO", "II"), "UH": ("UU",), "UW": ("UU",)}


EN_CONSONANTS = {"B": "PP", "M": "PP", "P": "PP", "F": "FF", "V": "FF", "TH": "TH", "DH": "TH", "T": "DD", "D": "DD",
                 "N": "DD", "L": "DD", "K": "KK", "G": "KK", "NG": "KK", "HH": "KK", "CH": "CH", "JH": "CH", "SH": "CH",
                 "ZH": "CH", "S": "SS", "Z": "SS", "R": "RR", "W": "UU", "Y": "II"}


ZH_INITIALS = {"b": "PP", "p": "PP", "m": "PP", "f": "FF", "d": "DD", "t": "DD", "n": "DD", "l": "DD", "g": "KK",
               "k": "KK", "h": "KK", "j": "SS", "q": "SS", "x": "SS", "zh": "CH", "ch": "CH", "sh": "CH", "r": "RR",
               "z": "SS", "c": "SS", "s": "SS"}


ZH_VOWELS = {"a": "AA", "o": "OO", "e": "EE", "i": "II", "u": "UU", "v": "UU", "ü": "UU"}


ES_VOWELS = {"a": "AA", "e": "EE", "i": "II", "o": "OO", "u": "UU", "ü": "UU"}


ES_CONSONANTS = {"b": "PP", "v": "PP", "p": "PP", "m": "PP", "f": "FF", "t": "DD", "d": "DD", "n": "DD", "l": "DD",
                 "ñ": "DD", "s": "SS", "z": "SS", "j": "KK", "k": "KK", "g": "KK", "w": "UU", "r": "RR"}


ES_ALPHABET = set("abcdefghijklmnopqrstuvwxyzñü")


LETTER_VISEMES = {"a": "AA", "e": "EE", "i": "II", "o": "OO", "u": "UU", "b": "PP", "m": "PP", "p": "PP", "f": "FF",
                  "v": "FF", "d": "DD", "l": "DD", "n": "DD", "t": "DD", "c": "KK", "g": "KK", "h": "KK", "k": "KK",
                  "q": "KK", "x": "KK", "j": "CH", "s": "SS", "z": "SS", "r": "RR", "w": "UU", "y": "II"}


LETTER_VOWELS = set("aeiou")


def _vowel(names, method):
    k = len(names)
    if (VOWEL_WEIGHT * _SCALE) % k:
        raise ValueError("a vowel split over %d visemes" % k)
    return [(V[n], VOWEL_WEIGHT * _SCALE // k, method) for n in names]


def _consonant(name, method):
    return [(V[name], CONSONANT_WEIGHT * _SCALE, method)]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _anyascii_units(s: str) -> list:
    out = []
    for ch in _anyascii()(s).lower():
        if ch in LETTER_VOWELS:
            out += _vowel((LETTER_VISEMES[ch],), "anyascii")
        elif ch in LETTER_VISEMES:
            out += _consonant(LETTER_VISEMES[ch], "anyascii")
    return out


@functools.lru_cache(maxsize=1)
def _g2p_en():
    """(CMUdict dict, g2p_en G2p, normalize_numbers); nltk data from NLTK_DATA, never downloaded here."""
    import nltk
    if os.path.isdir(NLTK_DATA) and NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA)
    for res in ("corpora/cmudict.zip", "taggers/averaged_perceptron_tagger.zip"):
        try:
            nltk.data.find(res)
        except LookupError:
            raise RuntimeError("nltk data %s missing (looked in %s); g2p_en would download it on import" % (res, NLTK_DATA))
    from g2p_en import G2p
    from g2p_en.expand import normalize_numbers
    g = G2p()
    return g.cmu, g, normalize_numbers


def _en_phones(phones, method):
    out = []
    for p in phones:
        p = re.sub(r"\d", "", p)
        if p in EN_VOWELS:
            out += _vowel(EN_VOWELS[p], method)
        elif p in EN_CONSONANTS:
            out += _consonant(EN_CONSONANTS[p], method)
    return out


def _en_units(w: str) -> tuple:
    """(units, n tokens, n OOV tokens)."""
    cmu, g2p, normalize_numbers = _g2p_en()
    text = _strip_accents(normalize_numbers(w)).lower()
    if any(c.isalpha() and not "a" <= c <= "z" for c in text):
        return _anyascii_units(w), 0, 0
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", text)
    out, oov = [], 0
    for tok in tokens:
        if tok in cmu:
            out += _en_phones(cmu[tok][0], "cmudict")
        else:
            oov += 1
            out += _en_phones(g2p.predict(tok), "g2p_en")
    if not tokens:
        out = _anyascii_units(w)
    return out, len(tokens), oov


_NONHAN = "\x00"


def _zh_units(w: str) -> list:
    from pypinyin import lazy_pinyin
    from pypinyin.contrib.tone_convert import to_finals, to_initials
    out = []
    for syl in lazy_pinyin(w, errors=lambda chunk: [_NONHAN + chunk]):
        if syl.startswith(_NONHAN):
            out += _anyascii_units(syl[1:])
            continue
        ini, fin = to_initials(syl, strict=True), to_finals(syl, strict=True)
        if ini:
            out += _consonant(ZH_INITIALS[ini], "pypinyin")
        if fin == "er":
            out += _vowel(("RR",), "pypinyin")
            continue
        coda = "ng" if fin.endswith("ng") else "n" if fin.endswith("n") else ""
        core = fin[:len(fin) - len(coda)]
        names = [ZH_VOWELS[c] for c in core if c in ZH_VOWELS]
        if names:
            out += _vowel(names, "pypinyin")
        if coda:
            out += _consonant("KK" if coda == "ng" else "DD", "pypinyin")
    return out


def _es_units(w: str) -> list:
    s = w.lower().translate(str.maketrans("áéíóú", "aeiou"))
    s = unicodedata.normalize("NFC", s)
    if any(c.isalpha() and c not in ES_ALPHABET for c in s):
        return _anyascii_units(w)
    s = "".join(c for c in s if c in ES_ALPHABET)
    out, i, m = [], 0, "es_rules"
    while i < len(s):
        c, nx = s[i], s[i + 1] if i + 1 < len(s) else ""
        step = 1
        if c in ES_VOWELS:
            out += _vowel((ES_VOWELS[c],), m)
        elif c == "c":
            if nx == "h":
                out, step = out + _consonant("CH", m), 2
            else:
                out += _consonant("SS" if nx in ("e", "i") else "KK", m)
        elif c == "l" and nx == "l":
            out, step = out + _consonant("II", m), 2
        elif c == "r" and nx == "r":
            out, step = out + _consonant("RR", m), 2
        elif c == "q" and nx == "u":
            out, step = out + _consonant("KK", m), 2
        elif c == "g" and nx == "u" and i + 2 < len(s) and s[i + 2] in ("e", "i"):
            out, step = out + _consonant("KK", m), 2
        elif c == "h":
            pass
        elif c == "y":
            out += _vowel(("II",), m) if (not nx or nx not in ES_VOWELS) else _consonant("II", m)
        elif c == "x":
            out += _consonant("KK", m) + _consonant("SS", m)
        elif c in ES_CONSONANTS:
            out += _consonant(ES_CONSONANTS[c], m)
        i += step
    return out


@functools.lru_cache(maxsize=None)
def _word(w: str, language) -> tuple:
    """(units tuple of (viseme id, weight, method), n en tokens, n en OOV tokens) of one word."""
    tok = oov = 0
    if language == "en":
        units, tok, oov = _en_units(w)
    elif language == "zh":
        units = _zh_units(w)
    elif language == "es":
        units = _es_units(w)
    else:
        units = _anyascii_units(w)
    if not units:
        units = [(VISEME_OTHER, CONSONANT_WEIGHT * _SCALE, "none")]
    return tuple(units), tok, oov


def allocate(units, n: int) -> np.ndarray:
    """Unit index of each of the n frames of a word (module doc: frame centre in the cumulative-weight interval)."""
    cum = np.cumsum([u[1] for u in units])
    centre2 = (2 * np.arange(n) + 1) * int(cum[-1])  # 2 n x frame centre
    return np.searchsorted(2 * n * cum, centre2, side="right")


def viseme_features(conditioning: dict, n_frames: int, hz: int = HZ) -> dict:
    """Per-frame viseme features of one load_conditioning_v2 track (words + language):
        viseme        (T,) int16    VISEMES id, 0 = sil off words
        boundary      (T, 3) float32 in_word, word_onset, word_offset (as the timing columns 0-2)
        viseme_onset  (T,) float32  1 on the first frame of each viseme run inside a word (incl. the word onset)
        g2p           (T,) bool     the frame's viseme came from the language's g2p (G2P_METHODS), not the fallback
    """
    if hz != HZ:
        raise ValueError("content-text words are on the %d Hz grid, got hz=%r" % (HZ, hz))
    if "language" not in conditioning:
        raise KeyError("the viseme path needs the text-level `language` (load_conditioning_v2)")
    language = conditioning["language"]
    vis = np.zeros(n_frames, dtype=np.int16)
    bnd = np.zeros((n_frames, 3), dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    g2p = np.zeros(n_frames, dtype=bool)
    words = conditioning["words"]
    spans = [(w["start_frame"], w["end_frame"]) for w in words]
    ends = [-1] + [e for _, e in spans]
    if not all(p < s <= e < n_frames for p, (s, e) in zip(ends, spans)):
        raise ValueError("words are not ordered, non-overlapping frames inside [0, %d)" % n_frames)
    for w, (s, e) in zip(words, spans):
        units = _word(w["w"], language)[0]
        k = allocate(units, e - s + 1)
        vis[s:e + 1] = [units[j][0] for j in k]
        g2p[s:e + 1] = [units[j][2] in G2P_METHODS for j in k]
        onset[s:e + 1] = np.concatenate([[1], np.diff(k) != 0]).astype(np.float32)
        bnd[s:e + 1, 0] = 1
        bnd[s, 1] = 1
        bnd[e, 2] = 1
    return {"viseme": vis, "boundary": bnd, "viseme_onset": onset, "g2p": g2p}


HZ_textfeat = HZ


TRAIN_RATE = 3.31  # words/s, the TRAIN mean (design §1.1)


SENTENCE_GAP_FRAMES = 13  # next sentence starts >= previous retimed end + 13 (a 12-frame pause)


def load_track_text(clip_id: str, track_id: str, n_frames: int, source: dict, lexical: str = "letters") -> dict:
    """The load_conditioning track of (clip_id, track_id), checked against the source's settings sha and T
    (lexical="viseme_v2": the load_conditioning_v2 track, i.e. + `language`)."""
    if lexical not in LEXICAL:
        raise ValueError("lexical must be one of %s" % (LEXICAL,))
    load = load_conditioning_v2 if lexical == "viseme_v2" else load_conditioning
    doc = load(clip_id, source["dir"])
    if source.get("settings_sha256") and doc["settings_sha256"] != source["settings_sha256"]:
        raise ValueError("%s: settings_sha256 %s != %s (%s)" % (clip_id, doc["settings_sha256"],
                                                               source["settings_sha256"], source["version"]))
    for t in doc["tracks"]:
        if t["track_id"] == track_id:
            if t["n_frames"] != n_frames:
                raise ValueError("%s: content text has %d frames, the L2 header %d" % (track_id, t["n_frames"], n_frames))
            return t
    raise KeyError("no content-text track %s in %s" % (track_id, clip_id))


def _round(v: float) -> int:
    return int(math.floor(v + 0.5))


def retime_words(words: list, n_frames: int, rate: float = TRAIN_RATE, gap: int = SENTENCE_GAP_FRAMES) -> tuple:
    """(retimed conditioning track fields {"words", "marks", "text"}, n_dropped) of frame words (module doc, m6)."""
    sentences, cur = [], []
    for w in words:
        cur.append(w)
        if w["w"].endswith(SENTENCE_END):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    out, dropped, prev_end = [], 0, None
    step = HZ_textfeat / float(rate)
    for sent in sentences:
        start = sent[0]["start_frame"] if prev_end is None else max(sent[0]["start_frame"], prev_end + gap)
        for k, w in enumerate(sent):
            s = start + _round(k * step)
            e = max(start + _round((k + 1) * step) - 1, s)
            if s >= n_frames:
                dropped += 1
                continue
            out.append({"w": w["w"], "start_frame": int(s), "end_frame": int(min(e, n_frames - 1))})
            prev_end = out[-1]["end_frame"]
        if prev_end is None:  # every word of the first sentence dropped (cannot happen with start < T)
            prev_end = start - 1
    marks = derive_marks(out)
    return {"words": out, "marks": marks, "text": render(out, marks)}, dropped


SCHEMA = "content_text/1.0"


PAUSE_MIN_SECONDS = 0.25


PAUSE_MIN_FRAMES = int(math.ceil(PAUSE_MIN_SECONDS * HZ))


MARK_KINDS = ("pause", "sentence_end", "punct")


WORD_KEYS = ("w", "start_frame", "end_frame", "prob")


DOC_KEYS = ("schema", "clip_id", "split", "hz", "settings_sha256", "tracks")


TRACK_KEYS = ("track_id", "segment_id", "track_start", "time_sec", "audio_status", "audio_valid_share", "n_frames",
              "status", "language", "language_prob", "words", "n_words_dropped", "marks", "text")


TRACK_KEYS_OPTIONAL = ("n_tokens_qwen", "silent")  # n_tokens_qwen: added by asr_transcribe.build after validation;


CONDITIONING_DOC_KEYS = ("schema", "clip_id", "hz", "settings_sha256", "tracks")


CONDITIONING_TRACK_KEYS = ("track_id", "segment_id", "track_start", "time_sec", "n_frames", "words", "marks", "text")


CONDITIONING_WORD_KEYS = ("w", "start_frame", "end_frame")


CONDITIONING_TRACK_KEYS_V2 = CONDITIONING_TRACK_KEYS + ("language",)


TRACK_STATUS = ("ok", "no_speech", "no_audio", "low_audio")


SPLIT_DIRS = ("train", "val", "test")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def derive_marks(words: list) -> list:
    """Frame words -> the allowed marks, in frame order (pause between i and i+1 after word i's own marks)."""
    marks = []
    for i, w in enumerate(words):
        text = w["w"]
        lead, tail = 0, len(text)
        while lead < tail and _is_punct(text[lead]):
            lead += 1
        while tail > lead and _is_punct(text[tail - 1]):
            tail -= 1
        for ch in text[:lead]:
            marks.append({"kind": "punct", "char": ch, "word": i, "side": "before", "frame": w["start_frame"]})
        for ch in text[tail:]:
            marks.append({"kind": "punct", "char": ch, "word": i, "side": "after", "frame": w["end_frame"]})
        if text.endswith(SENTENCE_END):
            marks.append({"kind": "sentence_end", "word": i, "frame": w["end_frame"]})
        if i + 1 < len(words):
            gap = words[i + 1]["start_frame"] - w["end_frame"] - 1
            if gap >= PAUSE_MIN_FRAMES:
                marks.append({"kind": "pause", "after_word": i, "start_frame": w["end_frame"] + 1,
                              "end_frame": words[i + 1]["start_frame"] - 1, "n_frames": gap})
    return marks


def render(words: list, marks: list) -> str:
    """The plain-text rendering (module doc); '<no speech>' when there are no words."""
    if not words:
        return "<no speech>"
    pause_after = {m["after_word"]: m["n_frames"] for m in marks if m["kind"] == "pause"}
    ends = {m["word"] for m in marks if m["kind"] == "sentence_end"}
    parts = []
    for i, w in enumerate(words):
        parts.append("[%.2f-%.2f] %s" % (w["start_frame"] / HZ, (w["end_frame"] + 1) / HZ, w["w"]))
        if i in pause_after:
            parts.append(" <pause %.2f>" % (pause_after[i] / HZ))
        if i + 1 < len(words):
            parts.append("\n" if i in ends else " ")
    return "".join(parts)


def validate(doc: dict) -> None:
    """Raise ValueError unless doc is a well-formed content-text clip json (frames, order, allowed keys / marks)."""
    def need(cond, msg):
        if not cond:
            raise ValueError("%s: %s" % (doc.get("clip_id"), msg))
    need(doc.get("schema") == SCHEMA, "schema != %s" % SCHEMA)
    need(doc.get("hz") == HZ, "hz != %d" % HZ)
    need(isinstance(doc.get("clip_id"), str) and isinstance(doc.get("tracks"), list) and doc["tracks"], "no tracks")
    need(set(doc) == set(DOC_KEYS), "clip keys %s" % sorted(set(doc) ^ set(DOC_KEYS)))
    for t in doc["tracks"]:
        need(set(TRACK_KEYS) <= set(t) <= set(TRACK_KEYS + TRACK_KEYS_OPTIONAL),
             "track keys: missing %s, not allowed %s" % (sorted(set(TRACK_KEYS) - set(t)),
                                                        sorted(set(t) - set(TRACK_KEYS + TRACK_KEYS_OPTIONAL))))
        n = t["n_frames"]
        need(t["status"] in TRACK_STATUS, "status %r" % t["status"])
        need(t["status"] != "ok" or t["words"], "status ok without words")
        if SILENT in t:
            need(isinstance(t[SILENT], bool), "silent must be true / false")
            need(not t[SILENT] or (not t["words"] and t["status"] == "no_speech"),
                 "silent: true needs a no_speech track without words")
        prev = -1
        for w in t["words"]:
            need(tuple(sorted(w)) == tuple(sorted(WORD_KEYS)), "word keys %s" % sorted(w))
            need(0 <= w["start_frame"] <= w["end_frame"] < n, "word frames outside [0, %d): %s" % (n, w))
            need(w["start_frame"] > prev, "words overlap or are not monotone at %s" % w)
            need(0.0 <= w["prob"] <= 1.0 and w["w"].strip() == w["w"] and w["w"], "word %s" % w)
            prev = w["end_frame"]
        for m in t["marks"]:
            need(m.get("kind") in MARK_KINDS, "mark kind %r not allowed" % m.get("kind"))
        need(t["marks"] == derive_marks(t["words"]), "marks are not the ones derived from the words")
        need(t["text"] == render(t["words"], t["marks"]), "text is not the rendering of the words")


def clip_path(clip_id: str, transcript_dir: str) -> str:
    """<dir>/<clip>.json, else <dir>/{train,val,test}/<clip>.json."""
    for sub in ("",) + SPLIT_DIRS:
        path = os.path.join(transcript_dir, sub, clip_id + ".json")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("no content text for %s under %s" % (clip_id, transcript_dir))


def load_content_text(clip_id: str, transcript_dir: str) -> dict:
    """The validated content-text json of one clip. Takes ONLY a clip id and the transcript dir (§5: no tensor / L1)."""
    with open(clip_path(clip_id, transcript_dir), encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("clip_id") != clip_id:
        raise ValueError("%s holds clip %r" % (clip_id, doc.get("clip_id")))
    validate(doc)
    return doc


def load_conditioning(clip_id: str, transcript_dir: str) -> dict:
    """THE entry point for G-diff conditioning: load_content_text reduced to CONDITIONING_*_KEYS, i.e. words (w,
    start_frame, end_frame), marks, text and the track structure; no prob / language_prob / audio_valid_share."""
    doc = load_content_text(clip_id, transcript_dir)
    out = {k: doc[k] for k in CONDITIONING_DOC_KEYS if k != "tracks"}
    out["tracks"] = [{**{k: t[k] for k in CONDITIONING_TRACK_KEYS if k != "words"},
                      "words": [{k: w[k] for k in CONDITIONING_WORD_KEYS} for w in t["words"]]} for t in doc["tracks"]]
    return out


def load_conditioning_v2(clip_id: str, transcript_dir: str) -> dict:
    """load_conditioning + the track's `language` (CONDITIONING_TRACK_KEYS_V2), for the opt-in viseme path only; + the
    optional `silent` flag (SILENT) where the track carries it."""
    doc = load_content_text(clip_id, transcript_dir)
    out = {k: doc[k] for k in CONDITIONING_DOC_KEYS if k != "tracks"}
    out["tracks"] = [{**{k: t[k] for k in CONDITIONING_TRACK_KEYS_V2 if k != "words"},
                      **({SILENT: t[SILENT]} if SILENT in t else {}),
                      "words": [{k: w[k] for k in CONDITIONING_WORD_KEYS} for w in t["words"]]} for t in doc["tracks"]]
    return out


VERSION = "viseme/1.0"
