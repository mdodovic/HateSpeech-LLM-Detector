"""
Base LLM Detector for Hate Speech Detection (Ollama backend)
"""

import re
import os
from pathlib import Path
import requests
from typing import List, Dict, Tuple


class LLMDetector:
    """Base class for hate speech detection using local Ollama service"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", default_temperature: float = 0.1, default_max_tokens: int = 1024):
        """
        Initialize the LLM detector (Ollama)

        Args:
            model_name: Ollama model tag (e.g., "llama3.2:3b", "phi3:mini", "mistral:7b")
            base_url: Base URL of the local Ollama server
            default_temperature: Default sampling temperature
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.max_tokens = default_max_tokens
        self._session = requests.Session()

    # --- Internal helpers ---
    def _clean_content(self, text: str) -> str:
        # Remove possible reasoning tags and trim
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _token_count(self, text: str) -> int:
        # Lightweight approximation of token count
        # Splits on words and punctuation to better approximate LLM tokens
        return len(re.findall(r"\w+|\S", text))

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        # Convert chat messages to a single prompt for /api/generate
        parts = []
        for m in messages:
            role = m.get("role", "user").strip()
            content = m.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _post(self, url: str, payload: Dict) -> requests.Response:
        return self._session.post(url, json=payload)

    def _extract_answer_from_thinking(self, text: str) -> str:
        """Best-effort extraction when reasoning models return content in 'thinking'.

        Tries to pull concise answer segments like 'Kategorija:' / 'Podkategorija:'
        or a DA/NE line for binary decisions. Falls back to cleaned text.
        """
        cleaned = self._clean_content(text)
        # Prefer explicit labeled lines if present
        m_cat = re.search(r"(?im)^\s*kategorija\s*:\s*(.+)$", cleaned)
        if m_cat:
            part = m_cat.group(0).strip()
            m_sub = re.search(r"(?im)^\s*podkategorija\s*:\s*(.+)$", cleaned)
            if m_sub:
                part = part + "\n" + m_sub.group(0).strip()
            return part
        # Try to return the first line containing DA/NE
        for ln in cleaned.splitlines():
            if re.search(r"\b(da|ne)\b", ln.strip().lower()):
                return ln.strip()
        # Fallback to the last 1-2 lines which often contain conclusion
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if lines:
            return "\n".join(lines[-2:]) if len(lines) > 1 else lines[-1]
        return cleaned.strip()

    def _chat(self, messages: List[Dict[str, str]], num_predict: int, temperature: float) -> str:
        # Try /api/chat first
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        chat_url = f"{self.base_url}/api/chat"
        resp = self._post(chat_url, chat_payload)
        if resp.status_code in (404, 405, 501):
            # Fallback to /api/generate (older Ollama versions)
            prompt = self._to_prompt(messages)
            gen_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            }
            gen_url = f"{self.base_url}/api/generate"
            gen_resp = self._post(gen_url, gen_payload)

            if gen_resp.status_code == 404 and "localhost" in self.base_url:
                # Try 127.0.0.1 fallback
                gen_url = gen_url.replace("localhost", "127.0.0.1")
                gen_resp = self._post(gen_url, gen_payload)

            gen_resp.raise_for_status()
            data = gen_resp.json()
            content = data.get("response", "")
            return self._clean_content(content)

        if resp.status_code == 404 and "localhost" in self.base_url:
            # Try 127.0.0.1 for /api/chat
            chat_url = chat_url.replace("localhost", "127.0.0.1")
            resp = self._post(chat_url, chat_payload)

        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("message") or {})
        content = (msg.get("content") or data.get("response") or "")

        if not content:
            # Reasoning models (e.g., deepseek-r1) may place output into 'thinking'
            reasoning = msg.get("reasoning") or msg.get("thinking") or data.get("thinking") or ""
            if reasoning:
                content = self._extract_answer_from_thinking(reasoning)
        return self._clean_content(content)

    # --- Public API (compatible with previous HF-based version) ---
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        messages = [
            {
                "role": "system",
                "content": "Vi ste pomoćnik za detekciju govora mržnje. Odgovarajte sažeto i tačno, prateći instrukcije.",
            },
            {"role": "user", "content": prompt},
        ]
        temp = temperature if temperature is not None else self.default_temperature
        return self._chat(messages, num_predict=max_new_tokens, temperature=temp)

    def detect_hate_speech_binary(self, text: str) -> bool:
        """
        Zadatak 1: Utvrdi da li tekst sadrži govor mržnje (binarna klasifikacija)
        Vraća: (sadrži_govor_mržnje)
        """
        # Load prompt relative to this file to avoid CWD issues
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_path = prompts_dir / "detect.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()

        prompt = prompt.format(text=text)

        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)
        contains_hate = False
        first_tokens = re.findall(r"\b\w+\b", response.lower())[:2]
        if any(tok in {"da"} for tok in first_tokens):
            contains_hate = True
        return contains_hate

    def extract_hate_speech_sentences(self, text: str) -> Tuple[List[str], int, int]:
        """
        Zadatak 2: Izdvoj rečenice koje sadrže govor mržnje i izračunaj pokrivenost tokena
        Vraća: (rečenice, pokriveni_tokeni, ukupno_tokena)
        """
        prompt = (
            "U datom tekstu identifikuj i izdvoj ISKLJUČIVO rečenice koje sadrže govor mržnje.\n"
            "Svaku rečenicu navedi u posebnom redu. Ako govor mržnje nije prisutan, odgovori sa \"NEMA\".\n\n"
            f'Tekst: "{text}"\n\n'
            "Rečenice sa govorom mržnje:"
        )

        response = self.generate_response(prompt, max_new_tokens=300, temperature=self.default_temperature)
        
        normalized = response.strip().lower()
        if normalized == "nema" or normalized.startswith("nema"):
            hate_sentences: List[str] = []
        else:
            raw_lines = re.split(r"[\n;]+", response)
            candidates = [ln.strip(" -•\t") for ln in raw_lines if ln.strip()]
            hate_sentences = [s for s in candidates if len(s) > 3]

        total_tokens = self._token_count(text)
        tokens_covered = self._token_count(" ".join(hate_sentences)) if hate_sentences else 0
        return hate_sentences, tokens_covered, total_tokens

    def categorize_hate_speech(self, text: str, categories_prompt: str) -> Tuple[int, str]:
        """
        Zadatak 3: Klasifikuj govor mržnje u unapred definisane kategorije (0–7)
        Vraća: (kategorija, podkategorija_kod)
        """
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_path = prompts_dir / "classify.txt"
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read().strip()  

        prompt = prompt.format(text=text, categories_prompt=categories_prompt)

        response = self.generate_response(prompt, max_new_tokens=self.max_tokens, temperature=self.default_temperature)
        # Ekstrakcija kategorije i podkategorije iz odgovora
        category = 0
        subcategory = ""

        # Pokušaj da pročitaš striktno iz polja 'Kategorija:' i 'Podkategorija:' ako postoje
        m_cat = re.search(r"(?i)kategorija\s*:\s*([0-7])\b", response)
        if m_cat:
            category = int(m_cat.group(1))
        else:
            # Fallback: uzmi prvi broj 0-7 gde god da se pojavi
            m = re.search(r"\b([0-7])\b", response)
            if m:
                category = int(m.group(1))

        # Podkategorija: preferiraj eksplicitno polje, npr. '3b'
        m_sub = re.search(r"(?i)podkategorija\s*:\s*([0-7][a-z])\b", response)
        if m_sub:
            subcategory = m_sub.group(1).lower()
        else:
            # Fallback: traži bilo koji kod poput '3b' negde u odgovoru
            m2 = re.search(r"\b([0-7][a-z])\b", response, flags=re.IGNORECASE)
            if m2:
                subcategory = m2.group(1).lower()

        return category, subcategory

    def analyze_text_complete(self, text: str, categories_prompt: str) -> Dict:
        """
        Perform all three tasks on a single text
        Returns: result dict
        """
        has_hate = self.detect_hate_speech_binary(text)
        binary_explanation = ""
        hate_sentences, tokens_covered, total_tokens = self.extract_hate_speech_sentences(text)
        category, category_subcategory = self.categorize_hate_speech(text, categories_prompt)
        return {
            "text": text,
            "has_hate_speech": has_hate,
            "binary_explanation": binary_explanation,
            "hate_sentences": hate_sentences,
            "tokens_covered": tokens_covered,
            "total_tokens": total_tokens,
            "token_coverage_ratio": (tokens_covered / total_tokens) if total_tokens > 0 else 0.0,
            "category": category,
            "category_subcategory": category_subcategory,
        }
