import regex as re
from typing import List, Dict, Set, Tuple

#for near misses and stuff
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import torch


# Regex patterns for transcript/OCR hits
PATTERNS = {
    "risk_free": re.compile(r"\b(risk[-\s]?free|no\s*risk)\b", re.I),
    "guaranteed": re.compile(r"\b(guaranteed\s*(win|profit)|can('?|no)t\s*lose|sure\s*bet|lock\s*of\s*the\s*day)\b", re.I),
    "free_but_risky": re.compile(r"\bfree\b", re.I),
    "chasing_losses": re.compile(r"\b(chase\s*loss(es)?|make\s*it\s*all\s*back|win\s*it\s*back|double\s*down\s*(your)?\s*loss(es)?)\b", re.I),
    "solve_financial_problems": re.compile(r"\b(pay\s*rent|bail\s*money|cover\s*fines|fix\s*debt|pay\s*off\s*(debt|loans))\b", re.I),
    "misrep_odds": re.compile(r"\b(guaranteed\s*streak|higher\s*odds\s*guarantee\s*wins|rigged\s*to\s*win)\b", re.I),
    "promo": re.compile(r"\b(code|ref(erral)?\s*code|promo\s*code|link\s*in\s*bio)\b", re.I),
    "vpn_proxy": re.compile(r"\b(use\s*a?\s*vpn|spoof\s*location|change\s*your\s*ip)\b", re.I),
    "youth_context": re.compile(r"\b(high\s*school|prom|teen(ager)?s?)\b", re.I),
    "college_cues": re.compile(r"\b(campus|dorm|university|college\s*(move[ -]?in|students?|freshman|frat|sorority)|NCAA)\b", re.I),
    "danger_social": re.compile(r"(rite\s*of\s*passage|trash(ed)?\s*(the|my)\s*room)", re.I),
    # RG markers
    "helpline": re.compile(r"(1[-\s]*800[-\s]*GAMBLER|GAMBLER)", re.I),
    "age21": re.compile(r"(21\+|must\s*be\s*21)", re.I),
    "promo_terms": re.compile(r"(terms|wager\s*requirement|min(imum)?\s*odds|playthrough|rollover)", re.I),
    "offshore_domain": re.compile(r"\b(?:stake\.com|roobet\.com|rainbet\.com|bovada\.(lv|ag)|[a-z0-9-]+\.(ag|lv|ph))\b", re.I),

    "wage_wager": re.compile(r"\b(paycheck|pay\s*check|my\s*(tips?|wages?|pay)|what\s+i\s+made?\s+(today|at\s*work)|my\s+shift\s+money)\b", re.I),
    "driving_phr": re.compile(r"\b(while\s*driving|drive\s+and\s+gambl(e|ing)|behind\s+the\s+wheel)\b", re.I),


}

def find_hits(text: str) -> Dict[str, List[Tuple[int, int]]]:
    hits = {}
    for name, pat in PATTERNS.items():
        spans = [m.span() for m in pat.finditer(text or "")]
        if spans:
            hits[name] = spans
    return hits
#Fuzzy hits for more complex phrases
FUZZY_PHRASES = {
    "chasing_losses": [
        "chase losses", "win it back", "make it all back", "double down to recover",
        "get it back", "earn it back", "regain my money", "recover my losses", "win back my losses", "win back what I lost", "win back what I lost", "get back what I lost", "get back what I lost", "earn back what I lost", "recover what I lost", "regain what I lost", "make it all back", "make it all back", "double down to recover", "double down to win it back"
    ],
    "wage_wager": [
        "gambling my paycheck", "gamble my wages", "bet my paycheck",
        "gambling what I make at work", "betting my tips", "put my shift money on it", "bet my wages", "bet my pay check",  "bet my paycheck", "gamble my pay check", "gamble my paycheck", "wager my pay check", "wager my paycheck"
    ],
    "risk_free": [
        "risk free", "no risk", "free risk", "strategy", "system", "no lose", "no lose strategy", "no lose system", "can't lose"
    ],
    "solve_financial_problems": [
        "pay rent with gambling", "cover fines with gambling", "bail money from gambling", "fix debt by gambling", "pay loans", "pay off debt", "pay debt", "pay bookie", "pay my debts", "pay my loans", "pay my loan", "pay my rent", "pay my fines", "pay off my debt", "pay off my loans", "pay off my loan", "pay off debt", "pay off loans", "pay off loan"
    ]
}

def fuzzy_hits(text: str, threshold: int = 85) -> Set[str]:
    fired = set()
    base = (text or "").lower()
    for key, phrases in FUZZY_PHRASES.items():
        for ph in phrases:
            if fuzz.partial_ratio(ph.lower(), base) >= threshold:
                fired.add(key)
                break
    return fired


# Embedding-based phrase matcher to capture semantic paraphrases
EMBED_MODEL: SentenceTransformer | None = None
PHRASE_EMBEDS: Dict[str, 'torch.Tensor'] = {}
EMBED_PHRASES = {
    "chasing_losses": [
        "try to win back losses",
        "recover what you lost",
        "double down to recover",
    ],
    "risk_free": [
        "can't lose",
        "no risk betting",
    ],
    "solve_financial_problems": [
        "pay off debt with gambling",
        "cover bills from bets",
    ],
}


def _get_embed_model() -> SentenceTransformer:
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return EMBED_MODEL


def embedding_hits(text: str, threshold: float = 0.7) -> Set[str]:
    """Return labels whose phrases are semantically similar to ``text``."""
    model = _get_embed_model()
    text_emb = model.encode(text, convert_to_tensor=True)
    fired: Set[str] = set()
    for label, phrases in EMBED_PHRASES.items():
        if label not in PHRASE_EMBEDS:
            PHRASE_EMBEDS[label] = model.encode(phrases, convert_to_tensor=True)
        sims = util.cos_sim(text_emb, PHRASE_EMBEDS[label])
        if float(sims.max()) >= threshold:
            fired.add(label)
    return fired
