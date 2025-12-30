"""
Definicije kategorija govora mržnje (srpski) + English label helpers
"""

HATE_SPEECH_CATEGORIES = {
    0: "Bez govora mržnje",
    1: "Rasna i etničko-nacionalna mržnja",
    2: "Verska mržnja",
    3: "Polna i rodna mržnja",
    4: "Mržnja zasnovana na fizičkim osobinama i zdravstvenom stanju",
    5: "Starosna i generacijska mržnja",
    6: "Socioekonomska mržnja",
    7: "Sportska i navijačka mržnja",
}

CATEGORY_DESCRIPTIONS = {
    0: "Tekst ne sadrži govor mržnje niti uvredljiv sadržaj usmeren na zaštićene grupe.",
    1: "Mržnja zasnovana na rasi/boji kože, etničkoj pripadnosti ili nacionalnosti/poreklu (uklj. migraciono, regionalno).",
    2: "Mržnja zasnovana na veri i veroispovesti (verska netrpeljivost, verska diskriminacija).",
    3: "Mržnja zasnovana na polu i/ili rodnom identitetu (seksizam; LGBTQ+ identiteti: homofobija, transfobija, queerfobija).",
    4: "Mržnja zasnovana na fizičkom izgledu i zdravstvenom stanju (diskriminacija na osnovu izgleda; bolest/invaliditet – ableizam).",
    5: "Mržnja zasnovana na uzrastu i generacijama (ageizam).",
    6: "Mržnja zasnovana na socioekonomskom statusu ili zanimanju (klasizam; stigmatizacija određenih profesija).",
    7: "Mržnja povezana sa sportom i navijačkim opredeljenjem (sportska netrpeljivost, huliganizam).",
}


# --- English labels (concise) ---
HATE_SPEECH_CATEGORIES_EN = {
    0: "No hate speech",
    1: "Racial and ethnic/national hate",
    2: "Religious hate",
    3: "Sex- and gender-based hate",
    4: "Physical traits and health-based hate",
    5: "Age- and generation-based hate",
    6: "Socioeconomic hate",
    7: "Sports and fan-based hate",
}

SUBCATEGORY_DESCRIPTIONS_EN = {
    1: {
        "1a": "Race / skin color",
        "1b": "Ethnic affiliation",
        "1c": "Nationality / origin (xenophobia)",
    },
    2: {
        "2": "Religion and faith",
    },
    3: {
        "3a": "Sex (sexism)",
        "3b": "LGBTQ+ identities",
    },
    4: {
        "4a": "Physical appearance",
        "4b": "Illness / disability",
    },
    5: {
        "5": "Age (ageism)",
    },
    6: {
        "6a": "Socioeconomic status / class",
        "6b": "Occupation / profession",
        "6c": "Political intolerance",
    },
    7: {
        "7": "Supporter affiliation",
    },
}

# Detaljnije potkategorije po vašoj specifikaciji
SUBCATEGORY_DESCRIPTIONS = {
    1: {
        "1a": "Rasa / boja kože → rasizam, kolorizam",
        "1b": "Etnička pripadnost → etnička diskriminacija, etnička mržnja",
        "1c": "Nacionalnost / poreklo (migraciono, regionalno) → ksenofobija",
    },
    2: {
        "2": "Vera i veroispovest → verska netrpeljivost, verska diskriminacija",
    },
    3: {
        "3a": "Pol → seksizam",
        "3b": "LGBTQ+ identiteti → homofobija, transfobija, queerfobija",
    },
    4: {
        "4a": "Fizički izgled → diskriminacija na osnovu izgleda (lookism)",
        "4b": "Bolest / invaliditet → ableizam",
    },
    5: {
        "5": "Uzrast (mladi, stari) → ageizam",
    },
    6: {
        "6a": "Socioekonomski status / klasa → klasizam",
        "6b": "Zanimanje / profesija → stigmatizacija određenih zanimanja",
        "6c": "Politička netrpeljivost → politička diskriminacija",
    },
    7: {
        "7": "Navijačko opredeljenje → sportska netrpeljivost, huliganizam",
    },
}


def get_category_prompt(include_subcategories: bool = True) -> str:
    """Generiše prompt sa kategorijama (i potkategorijama ako je uključeno)."""
    prompt = "Kategorije i podkategorije govora mržnje su sledeće:\n"
    for cat_id in sorted(HATE_SPEECH_CATEGORIES.keys()):
        prompt += f"{cat_id}: {HATE_SPEECH_CATEGORIES[cat_id]}\n   {CATEGORY_DESCRIPTIONS[cat_id]}\n"
        if include_subcategories and cat_id in SUBCATEGORY_DESCRIPTIONS:
            for code, desc in SUBCATEGORY_DESCRIPTIONS[cat_id].items():
                prompt += f"   - {code}: {desc}\n"
        prompt += "\n"
    return prompt


def code_to_label(code: str) -> str:
    """Return a concise human-readable label for a category/subcategory code.

    Examples:
      code_to_label("0")   -> "Bez govora mržnje"
      code_to_label("1")   -> "Rasna i etničko-nacionalna mržnja"
      code_to_label("1a")  -> "Rasa / boja kože"
      code_to_label("3b")  -> "LGBTQ+ identiteti"
      code_to_label("6c")  -> "Politička netrpeljivost"
    """
    if not isinstance(code, str):
        code = str(code or "")
    s = code.strip().lower()
    if s == "0":
        return HATE_SPEECH_CATEGORIES.get(0, "Bez govora mržnje")
    if s in {"u", "uvreda"}:
        return "Uvreda"
    # Subcategory like '3b'
    import re as _re
    m = _re.match(r"^([0-7])([a-z])$", s)
    if m:
        cid = int(m.group(1))
        subcode = s
        # If subcategory is 'u' (offense), label as Uvreda regardless of category
        if subcode.endswith("u"):
            return "Uvreda"
        submap = SUBCATEGORY_DESCRIPTIONS.get(cid, {})
        desc = submap.get(subcode)
        if isinstance(desc, str) and desc:
            # Extract label before '→' if present, else full string
            label = desc.split("→", 1)[0].strip()
            return label or desc
        return HATE_SPEECH_CATEGORIES.get(cid, f"Kategorija {cid}")
    # Top-level like '3'
    m2 = _re.match(r"^([0-7])$", s)
    if m2:
        cid = int(m2.group(1))
        # Some categories have top-level entries in SUBCATEGORY_DESCRIPTIONS (e.g., '2')
        submap = SUBCATEGORY_DESCRIPTIONS.get(cid, {})
        top_desc = submap.get(str(cid))
        if isinstance(top_desc, str) and top_desc:
            return top_desc.split("→", 1)[0].strip()
        return HATE_SPEECH_CATEGORIES.get(cid, f"Kategorija {cid}")
    # Fallback for unknown format
    return s or ""


def code_to_label_en(code: str) -> str:
    """Return a concise English label for a category/subcategory code.

    Examples:
      code_to_label_en("0")   -> "No hate speech"
      code_to_label_en("1")   -> "Racial and ethnic/national hate"
      code_to_label_en("1a")  -> "Race / skin color"
      code_to_label_en("3b")  -> "LGBTQ+ identities"
      code_to_label_en("6c")  -> "Political intolerance"
    """
    if not isinstance(code, str):
        code = str(code or "")
    s = code.strip().lower()
    if s == "0":
        return HATE_SPEECH_CATEGORIES_EN.get(0, "No hate speech")
    if s in {"u", "uvreda", "offense"}:
        return "Offense"
    import re as _re
    m = _re.match(r"^([0-7])([a-z])$", s)
    if m:
        cid = int(m.group(1))
        subcode = s
        if subcode.endswith("u"):
            return "Offense"
        submap = SUBCATEGORY_DESCRIPTIONS_EN.get(cid, {})
        desc = submap.get(subcode)
        if isinstance(desc, str) and desc:
            # For English we store concise labels directly
            return desc.strip()
        return HATE_SPEECH_CATEGORIES_EN.get(cid, f"Category {cid}")
    m2 = _re.match(r"^([0-7])$", s)
    if m2:
        cid = int(m2.group(1))
        submap = SUBCATEGORY_DESCRIPTIONS_EN.get(cid, {})
        top_desc = submap.get(str(cid))
        if isinstance(top_desc, str) and top_desc:
            return top_desc.strip()
        return HATE_SPEECH_CATEGORIES_EN.get(cid, f"Category {cid}")
    return s or ""
