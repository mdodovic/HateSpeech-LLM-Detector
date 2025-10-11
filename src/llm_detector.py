"""
Base LLM Detector for Hate Speech Detection
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re


class LLMDetector:
    """Base class for hate speech detection using LLMs"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize the LLM detector
        
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to run the model on ("cuda", "cpu", or "auto")
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if device == "auto" else None,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model = self.model.to(self.device)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    
    def detect_hate_speech_binary(self, text: str) -> Tuple[bool, str]:
        """
        Task 1: Detect if text contains hate speech (binary classification)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (contains_hate_speech: bool, explanation: str)
        """
        prompt = f"""Analyze the following text and determine if it contains hate speech.
Hate speech is language that attacks or uses pejorative or discriminatory language with reference to a person or group based on attributes such as race, ethnicity, gender, religion, sexual orientation, disability, or nationality.

Text: "{text}"

Answer with either "YES" or "NO" followed by a brief explanation.
Answer:"""
        
        response = self.generate_response(prompt, max_new_tokens=150)
        
        # Parse response
        contains_hate = "yes" in response.lower().split()[0:2]
        
        return contains_hate, response
    
    def extract_hate_speech_sentences(self, text: str) -> Tuple[List[str], int, int]:
        """
        Task 2: Extract sentences containing hate speech and count tokens
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (hate_sentences: List[str], tokens_covered: int, total_tokens: int)
        """
        prompt = f"""Given the following text, identify and extract ONLY the sentences that contain hate speech.
List each sentence on a new line. If no hate speech is present, respond with "NONE".

Text: "{text}"

Hate speech sentences:"""
        
        response = self.generate_response(prompt, max_new_tokens=300)
        
        # Parse extracted sentences
        if "none" in response.lower().strip():
            hate_sentences = []
        else:
            # Split by common delimiters
            sentences = [s.strip() for s in re.split(r'[;\n]', response) if s.strip()]
            # Filter out very short responses or artifacts
            hate_sentences = [s for s in sentences if len(s) > 10]
        
        # Calculate token coverage
        total_tokens = len(self.tokenizer.encode(text))
        if hate_sentences:
            hate_text = " ".join(hate_sentences)
            tokens_covered = len(self.tokenizer.encode(hate_text))
        else:
            tokens_covered = 0
        
        return hate_sentences, tokens_covered, total_tokens
    
    def categorize_hate_speech(self, text: str, categories_prompt: str) -> Tuple[int, str]:
        """
        Task 3: Categorize hate speech into predefined categories (0-7)
        
        Args:
            text: Input text to analyze
            categories_prompt: Description of categories
            
        Returns:
            Tuple of (category: int, explanation: str)
        """
        prompt = f"""{categories_prompt}

Analyze the following text and classify it into one of the categories above (0-7).
Respond with the category number followed by a brief explanation.

Text: "{text}"

Category:"""
        
        response = self.generate_response(prompt, max_new_tokens=200)
        
        # Extract category number
        category = 0  # Default to no hate speech
        match = re.search(r'\b([0-7])\b', response)
        if match:
            category = int(match.group(1))
        
        return category, response
    
    def analyze_text_complete(self, text: str, categories_prompt: str) -> Dict:
        """
        Perform all three tasks on a single text
        
        Args:
            text: Input text to analyze
            categories_prompt: Description of categories
            
        Returns:
            Dictionary with all results
        """
        # Task 1: Binary detection
        has_hate, binary_explanation = self.detect_hate_speech_binary(text)
        
        # Task 2: Extract sentences
        hate_sentences, tokens_covered, total_tokens = self.extract_hate_speech_sentences(text)
        
        # Task 3: Categorization
        category, category_explanation = self.categorize_hate_speech(text, categories_prompt)
        
        return {
            "text": text,
            "has_hate_speech": has_hate,
            "binary_explanation": binary_explanation,
            "hate_sentences": hate_sentences,
            "tokens_covered": tokens_covered,
            "total_tokens": total_tokens,
            "token_coverage_ratio": tokens_covered / total_tokens if total_tokens > 0 else 0,
            "category": category,
            "category_explanation": category_explanation
        }
