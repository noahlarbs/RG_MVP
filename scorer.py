from typing import Dict, Tuple, List

# Exposed weights so other modules (e.g., dataset labeling) know which flags exist
WEIGHTS: Dict[str, int] = {
    # Claims
    "risk_free": 30, "guaranteed": 25, "free_but_risky": 30,
    "chasing_losses": 18, "solve_financial_problems": 15, "misrep_odds": 12,
    # Age
    "youth_context": 22, "college_cues": 22, "under21_endorser": 25,
    # RG messaging
    "missing_helpline": 12, "missing_21plus": 10, "missing_terms": 10,
    # Offshore / availability
    "offshore_brand": 22, "vpn_proxy": 10, "unapproved_ref": 18,
    # Dangerous behavior
    "danger_driving": 15, "socially_irresponsible": 10, "wage_wager": 12,
    # Endorsements
    "undisclosed_affiliate": 12,
}

def score_clip(features: Dict) -> Tuple[int, Dict[str,int], List[Tuple[str,str]]]:
    weights = WEIGHTS

    flags = []
    s_cat = {"age":0,"claims":0,"rgmsg":0,"offshore":0,"danger":0,"endorse":0}

    phrases = features.get("phrases", set())
    ops = set(features.get("operators", []))

    # Claims
    if "risk_free" in phrases:
        flags.append(("claims","risk_free")); s_cat["claims"] += weights["risk_free"]
    if "guaranteed" in phrases:
        flags.append(("claims","guaranteed")); s_cat["claims"] += weights["guaranteed"]
    if "free_but_risky" in phrases and not features.get("has_promo_terms", False):
        flags.append(("claims","free_but_risky")); s_cat["claims"] += weights["free_but_risky"]
    if "chasing_losses" in phrases:
        flags.append(("claims","chasing_losses")); s_cat["claims"] += weights["chasing_losses"]
    if "solve_financial_problems" in phrases:
        flags.append(("claims","solve_financial_problems")); s_cat["claims"] += weights["solve_financial_problems"]
    if "misrep_odds" in phrases:
        flags.append(("claims","misrep_odds")); s_cat["claims"] += weights["misrep_odds"]
    if "wage_wager" in phrases:
        flags.append(("claims","wage_wager")); s_cat["claims"] += weights["wage_wager"]


    # Age
    if features.get("youth_context"): 
        flags.append(("age","youth_context")); s_cat["age"] += weights["youth_context"]
    if features.get("college_cues"): 
        flags.append(("age","college_cues")); s_cat["age"] += weights["college_cues"]
    if features.get("under21_endorser", False):
        flags.append(("age","under21_endorser")); s_cat["age"] += weights["under21_endorser"]

    # RG messaging (if promo/brand present)
    if (ops or ("promo" in phrases)) and not features.get("has_helpline"):
        flags.append(("rgmsg","missing_helpline")); s_cat["rgmsg"] += weights["missing_helpline"]
    if (ops or ("promo" in phrases)) and not features.get("has_21plus"):
        flags.append(("rgmsg","missing_21plus")); s_cat["rgmsg"] += weights["missing_21plus"]
    if "promo" in phrases and not features.get("has_promo_terms"):
        flags.append(("rgmsg","missing_terms")); s_cat["rgmsg"] += weights["missing_terms"]

    # Offshore / availability
    if ops & {"bovada","stake","roobet","rainbet","rollbit"}:
        flags.append(("offshore","offshore_brand")); s_cat["offshore"] += weights["offshore_brand"]
    if features.get("vpn_proxy"): 
        flags.append(("offshore","vpn_proxy")); s_cat["offshore"] += weights["vpn_proxy"]
    if features.get("unapproved_ref"): 
        flags.append(("offshore","unapproved_ref")); s_cat["offshore"] += weights["unapproved_ref"]

    # Dangerous behavior
    if features.get("danger_driving"):
        flags.append(("danger","danger_driving")); s_cat["danger"] += weights["danger_driving"]
    if features.get("socially_irresponsible"):
        flags.append(("danger","socially_irresponsible")); s_cat["danger"] += weights["socially_irresponsible"]

    # Endorsements
    if features.get("affiliate_undisclosed"):
        flags.append(("endorse","undisclosed_affiliate")); s_cat["endorse"] += weights["undisclosed_affiliate"]

    overall = min(100, sum(s_cat.values()))
    return overall, s_cat, flags
