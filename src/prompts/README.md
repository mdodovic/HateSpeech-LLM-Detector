# Prompts Directory

This directory contains prompts for the HateSpeech LLM Detector, organized by language.

## Structure

```
prompts/
├── serbian/          # Serbian language prompts (Srpski)
│   ├── system_prompt.txt
│   ├── classification_prompt.txt
│   └── analysis_prompt.txt
├── english/          # English language prompts
│   ├── system_prompt.txt
│   ├── classification_prompt.txt
│   └── analysis_prompt.txt
└── README.md
```

## Prompt Types

### system_prompt.txt
The system prompt that defines the role and context for the LLM. This sets the general behavior and expertise of the model for hate speech detection.

### classification_prompt.txt
A prompt for binary classification of text as hate speech or not. Returns a structured JSON response with:
- `is_hate_speech`: Boolean indicating if text contains hate speech
- `confidence`: Confidence score (0-1)
- `categories`: List of hate speech categories detected
- `explanation`: Brief explanation of the decision

### analysis_prompt.txt
A prompt for detailed analysis of potentially hateful content, providing:
- Hate speech determination
- Specific problematic elements
- Target groups
- Severity level
- Moderation recommendations

## Usage

To use these prompts in your code, load them using the prompt loader utility:

```python
from prompt_loader import load_prompt

# Load Serbian classification prompt
prompt = load_prompt('serbian', 'classification')
formatted_prompt = prompt.format(text="Sample text to analyze")

# Load English system prompt
system_prompt = load_prompt('english', 'system')
```

## Supported Languages

- **Serbian (Srpski)**: Full support for hate speech detection in Serbian language
- **English**: Full support for hate speech detection in English language

## Adding New Languages

To add support for a new language:

1. Create a new directory: `prompts/<language_name>/`
2. Create the three prompt files:
   - `system_prompt.txt`
   - `classification_prompt.txt`
   - `analysis_prompt.txt`
3. Translate the prompts appropriately for the target language

## Customization

You can customize these prompts based on your specific use case:
- Adjust the categories of hate speech to detect
- Modify the response format
- Add context-specific instructions
- Include examples for few-shot learning

## Notes

- Prompts use `{text}` as a placeholder for the input text to analyze
- The Serbian prompts are the primary focus of this project
- JSON format responses enable easy parsing and integration with applications
