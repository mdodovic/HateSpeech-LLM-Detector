# Hate Speech Categories

This document provides detailed descriptions of the 8 hate speech categories used in this framework.

## Category 0: No Hate Speech

**Description**: The text does not contain any hate speech or offensive content targeting protected groups.

**Examples**:
- "I love spending time with my friends from different cultures."
- "Everyone deserves equal rights and opportunities."
- "The weather is beautiful today."

**Key Indicators**:
- Neutral or positive language
- No targeting of protected groups
- Respectful discourse

---

## Category 1: Race/Ethnicity-based Hate Speech

**Description**: Hate speech targeting individuals or groups based on their race, ethnicity, or skin color.

**Examples**:
- Derogatory racial slurs
- Statements claiming racial superiority or inferiority
- Incitement to discrimination based on race

**Protected Groups**:
- Racial minorities
- Ethnic groups
- People of specific skin colors

**Key Indicators**:
- Racial slurs or stereotypes
- Claims of racial hierarchy
- Calls for racial segregation or discrimination

---

## Category 2: Religion-based Hate Speech

**Description**: Hate speech targeting individuals or groups based on their religious beliefs or practices.

**Examples**:
- Statements claiming all members of a religion are dangerous
- Calls to ban or restrict religious practices
- Dehumanizing language about religious groups

**Protected Groups**:
- Muslims, Christians, Jews, Hindus, Buddhists
- Members of any religious faith
- Atheists and agnostics

**Key Indicators**:
- Religious stereotypes
- Claims that a religion is inherently violent or evil
- Calls for religious persecution

---

## Category 3: Gender-based Hate Speech

**Description**: Hate speech targeting individuals based on their gender identity or expression.

**Examples**:
- Statements claiming one gender is inferior
- Derogatory language about women or men
- Attacks on transgender individuals

**Protected Groups**:
- Women
- Men
- Transgender individuals
- Non-binary individuals

**Key Indicators**:
- Gender-based slurs
- Claims of gender superiority/inferiority
- Denial of gender identity validity

---

## Category 4: Sexual Orientation-based Hate Speech

**Description**: Hate speech targeting individuals based on their sexual orientation or LGBTQ+ identity.

**Examples**:
- Homophobic slurs or language
- Claims that LGBTQ+ individuals are abnormal or immoral
- Calls to restrict LGBTQ+ rights

**Protected Groups**:
- Gay, lesbian, bisexual individuals
- LGBTQ+ community members
- Queer individuals

**Key Indicators**:
- Homophobic or transphobic slurs
- Claims that LGBTQ+ identities are "wrong" or "unnatural"
- Promotion of conversion therapy

---

## Category 5: Disability-based Hate Speech

**Description**: Hate speech targeting individuals with physical, mental, or developmental disabilities.

**Examples**:
- Derogatory terms for people with disabilities
- Statements claiming people with disabilities are burdens
- Calls to exclude people with disabilities from public spaces

**Protected Groups**:
- People with physical disabilities
- People with mental health conditions
- People with developmental disabilities
- Neurodiverse individuals

**Key Indicators**:
- Ableist slurs
- Claims that people with disabilities lack value
- Mocking of disability accommodations

---

## Category 6: Nationality-based Hate Speech

**Description**: Hate speech targeting individuals based on their country of origin or nationality.

**Examples**:
- Statements claiming all people from a country are criminals
- Xenophobic language about immigrants
- Calls for deportation based on nationality

**Protected Groups**:
- Immigrants
- Refugees
- Foreign nationals
- People based on country of origin

**Key Indicators**:
- Xenophobic language
- Stereotypes about nationalities
- Calls for immigration restrictions based on prejudice

---

## Category 7: Other Forms of Hate Speech

**Description**: Other forms of hate speech including targeting based on age, social class, or other characteristics not covered in categories 1-6.

**Examples**:
- Ageist discrimination (e.g., against elderly or youth)
- Classist hate speech
- Hate speech based on appearance
- Caste-based discrimination

**Protected Groups**:
- Elderly individuals
- Young people
- Lower socioeconomic classes
- People based on physical appearance

**Key Indicators**:
- Discriminatory language not covered by other categories
- Systematic dehumanization of groups
- Calls for exclusion or harm

---

## Annotation Guidelines

When categorizing hate speech:

1. **Identify the primary target**: If multiple groups are targeted, choose the most prominent one.

2. **Intent matters**: Consider whether the language is intended to harm, demean, or incite hatred.

3. **Context is important**: Some words may be hateful in one context but not another.

4. **Intersectionality**: If hate speech targets multiple categories, choose the most salient one.

5. **Severity**: All categories are equally serious; there is no hierarchy of hate speech severity.

## Common Edge Cases

### Critique vs. Hate Speech
- **Critique**: "I disagree with this religious practice because..."
- **Hate Speech**: "People of this religion are all evil."

### Humor vs. Hate Speech
- Jokes that demean or dehumanize protected groups are still hate speech
- Intent to "just joke" does not negate hateful content

### Reclaimed Language
- Some communities reclaim slurs used against them
- Context and speaker identity matter for these cases

### News Reporting
- Reporting on hate speech (quoting it) is different from producing hate speech
- Academic discussion of hate speech is not hate speech itself

## Quality Control

For reliable categorization:

1. **Multiple annotators**: Have at least 2-3 people categorize each text
2. **Inter-annotator agreement**: Measure agreement using Cohen's kappa or similar
3. **Clear guidelines**: Provide annotators with detailed examples
4. **Regular calibration**: Periodically review difficult cases as a team
5. **Cultural sensitivity**: Ensure annotators understand cultural context

## References and Resources

- [UN Strategy and Plan of Action on Hate Speech](https://www.un.org/en/genocideprevention/hate-speech-strategy.shtml)
- [UNESCO's Role in Combating Online Hate Speech](https://www.unesco.org/en/hate-speech)
- [EU Code of Conduct on Countering Illegal Hate Speech](https://ec.europa.eu/info/policies/justice-and-fundamental-rights/combatting-discrimination/racism-and-xenophobia/eu-code-conduct-countering-illegal-hate-speech-online_en)
