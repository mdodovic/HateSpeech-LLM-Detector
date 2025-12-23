"""
Definicije kategorija govora mržnje (srpski)
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
