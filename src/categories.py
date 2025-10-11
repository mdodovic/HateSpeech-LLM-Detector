"""
Hate Speech Categories Definition
"""

HATE_SPEECH_CATEGORIES = {
    0: "No hate speech",
    1: "Race/Ethnicity-based hate speech",
    2: "Religion-based hate speech",
    3: "Gender-based hate speech",
    4: "Sexual orientation-based hate speech",
    5: "Disability-based hate speech",
    6: "Nationality-based hate speech",
    7: "Other forms of hate speech"
}

CATEGORY_DESCRIPTIONS = {
    0: "The text does not contain any hate speech or offensive content targeting protected groups.",
    1: "Hate speech targeting individuals or groups based on their race, ethnicity, or skin color.",
    2: "Hate speech targeting individuals or groups based on their religious beliefs or practices.",
    3: "Hate speech targeting individuals based on their gender identity or expression.",
    4: "Hate speech targeting individuals based on their sexual orientation or LGBTQ+ identity.",
    5: "Hate speech targeting individuals with physical, mental, or developmental disabilities.",
    6: "Hate speech targeting individuals based on their country of origin or nationality.",
    7: "Other forms of hate speech including age, social class, or other characteristics not covered above."
}

def get_category_prompt():
    """Generate a prompt describing all categories for LLM"""
    prompt = "Hate Speech Categories:\n"
    for cat_id, description in CATEGORY_DESCRIPTIONS.items():
        prompt += f"{cat_id}: {HATE_SPEECH_CATEGORIES[cat_id]}\n   {description}\n\n"
    return prompt
