# Synthetic vs. Authentic Data Analysis

Input file: `data\access_paragraph_hate_speech_with_offenses.xlsx`
Final labels: `Annotator3 (Senior)` per sentence when present, otherwise `Annotator1`.
Sentence counts are based on annotation entries, matching the 8,029 labeled sentences.

## Subset Size
| Subset | Paragraphs | Sentences |
| --- | --- | --- |
| Synthetic | 51 | 267 |
| Authentic | 1300 | 7762 |

## Split Placement
PASS: synthetic paragraphs in held-out test split (ID <= 101) = 0

## Paragraph Length
| Subset | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| Synthetic | 5.235 | 4 | 4 | 12 |
| Authentic | 5.971 | 5.000 | 3 | 33 |

## Sentence-Level Class Distribution
Priority: hate if any 1-7 label is present, otherwise offensive if U is present, otherwise neutral.
| Subset | Class | Count | Percent |
| --- | --- | --- | --- |
| Synthetic | neutral | 74 | 27.715 |
| Synthetic | offensive | 34 | 12.734 |
| Synthetic | hate | 159 | 59.551 |
| Authentic | neutral | 4613 | 59.431 |
| Authentic | offensive | 616 | 7.936 |
| Authentic | hate | 2533 | 32.633 |

## Hate Category Distribution
Counts are label-level: each hate label in a multi-label sentence is counted separately.
Synthetic hate-label denominator = 169; authentic hate-label denominator = 2736.
| Subset | Category | Label count | Percent |
| --- | --- | --- | --- |
| Synthetic | 1 | 59 | 34.911 |
| Synthetic | 2 | 23 | 13.609 |
| Synthetic | 3 | 25 | 14.793 |
| Synthetic | 4 | 15 | 8.876 |
| Synthetic | 5 | 1 | 0.592 |
| Synthetic | 6 | 38 | 22.485 |
| Synthetic | 7 | 8 | 4.734 |
| Authentic | 1 | 871 | 31.835 |
| Authentic | 2 | 57 | 2.083 |
| Authentic | 3 | 430 | 15.716 |
| Authentic | 4 | 185 | 6.762 |
| Authentic | 5 | 69 | 2.522 |
| Authentic | 6 | 775 | 28.326 |
| Authentic | 7 | 349 | 12.756 |

## Synthetic Subcategory Distribution
| Subcategory | Label count | Percent |
| --- | --- | --- |
| 1a | 27 | 15.976 |
| 1b | 18 | 10.651 |
| 1c | 13 | 7.692 |
| 2 | 23 | 13.609 |
| 3a | 6 | 3.550 |
| 3b | 19 | 11.243 |
| 4a | 5 | 2.959 |
| 4b | 10 | 5.917 |
| 5 | 1 | 0.592 |
| 6a | 10 | 5.917 |
| 6b | 18 | 10.651 |
| 6c | 10 | 5.917 |
| 7 | 8 | 4.734 |

## Annotator1 vs Annotator2 Disagreement
| Subset | Compared sentences | Disagreements | Disagreement percent |
| --- | --- | --- | --- |
| Synthetic | 267 | 52 | 19.476 |
| Authentic | 7762 | 193 | 2.486 |
