#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JP-RU Historical Terms – автономный Gradio-интерфейс
Запустите:  python jp_ru_gradio.py
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import warnings
from typing import Dict, Optional

import gradio as gr
import requests
import wikipedia
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------------------------------------------------------ #
# 0. Отключаем шум
# ------------------------------------------------------------------ #
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
wikipedia.set_lang("ja")

# ------------------------------------------------------------------ #
# 1. Инициализация once-at-startup
# ------------------------------------------------------------------ #
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print("Загружаем модель… (~30-60 с на CPU, быстрее на GPU)")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mod = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
pipe = pipeline(
    "text-generation",
    model=mod,
    tokenizer=tok,
    max_new_tokens=300,
    temperature=0.05,
    return_full_text=False,
    pad_token_id=tok.eos_token_id
)
print("↗️  Модель готова")

# ------------------------------------------------------------------ #
# 2. Утилиты Wikidata / Википедия
# ------------------------------------------------------------------ #
HEADERS = {"User-Agent": "jp-ru-helper/1.0 (gradio)"}
session = requests.Session()
session.headers.update(HEADERS)

def get_qid_sitelinks(page_title: str, lang: str = "ja") -> Dict[str, str]:
    """Возвращает словарь {lang: title} для статьи page_title в Википедии языка lang."""
    # 1) QID
    api = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        resp = session.get(
            api,
            params={"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
                    "titles": page_title, "format": "json"},
            timeout=12
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        qid = next(iter(pages.values()), {}).get("pageprops", {}).get("wikibase_item")
        if not qid:
            return {}
    except Exception as e:
        print("⚠️ QID error:", e)
        return {}

    # 2) sitelinks
    try:
        resp2 = session.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetentities", "ids": qid, "props": "sitelinks", "format": "json"},
            timeout=12
        )
        resp2.raise_for_status()
        sl = resp2.json().get("entities", {}).get(qid, {}).get("sitelinks", {})
        return {s.replace("wiki", ""): data["title"] for s, data in sl.items() if s.endswith("wiki")}
    except Exception as e:
        print("⚠️ sitelinks error:", e)
        return {}

def ja2ru_wiki_page(ja_term: str) -> tuple[str, str]:
    """Возвращает (ru_title, ru_text) по японскому термину."""
    try:
        ja_page = wikipedia.page(ja_term, auto_suggest=False)
    except Exception as e:
        print("❌ ja-page not found:", e)
        return None, None

    sitelinks = get_qid_sitelinks(ja_page.title, lang="ja")
    ru_title = sitelinks.get("ru")
    if not ru_title:
        print("❌ ru-interwiki absent")
        return None, None

    wikipedia.set_lang("ru")
    try:
        ru_page = wikipedia.page(ru_title, auto_suggest=False)
    except Exception as e:
        print("❌ ru-page not found:", e)
        return None, None
    return ru_page.title, ru_page.content[:1_200]

# ------------------------------------------------------------------ #
# 3. Ядро агента
# ------------------------------------------------------------------ #
class Monitor:
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = self.end = time.time()
        self.llm_calls = self.wiki_calls = 0
monitor = Monitor()

def extract_term(text: str) -> str:
    """Извлекает японский термин из текста."""
    prompt = f"""<|im_start|>system
Ты извлекаешь японский термин из текста пользователя.
1. Ответь **точно теми же иероглифами / каной**, что есть во входе.
2. Не переводи, не транскрибируй, не добавляй пояснений.
3. Выведи **одну непрерывную строку** только из японских символов (кандзи, хирагана, катакана).
4. Если таких символов несколько — верни самую длинную непрерывную последовательность.
<|im_start|>user
{text}
<|im_start|>assistant"""
    monitor.llm_calls += 1
    out = pipe(prompt, max_new_tokens=50)[0]["generated_text"].strip()
    return out.splitlines()[0] if out else ""

def ask_agent(query: str) -> str:
    """Основной вызов агента для Gradio."""
    monitor.reset()
    monitor.start = time.time()

    term = extract_term(query)
    if not term:
        return "Не удалось выделить японский термин."

    ru_title, ru_text = ja2ru_wiki_page(term)
    if not ru_title:
        return "Русская статья не найдена, попробуйте переформулировать."

    final_prompt = f"""<|im_start|>system
Ты — помощник переводчика с японского на русский. Используя факты из русской Википедии (ниже), дай:
1) сообщи, что самый удачный вариант перевода на русский это - заголовок аналогичной статьи в русской Википедии: {ru_title};
2) дай дословный перевод японского термина {term} на русский;
3) дай краткую справку (1-2 предложения) о том, что это такое, используя факты из статьи.

Факты из Википедии:
{ru_text}
<|im_start|>user
Как перевести «{term}»?
<|im_start|>assistant"""
    monitor.llm_calls += 1
    answer = pipe(final_prompt, max_new_tokens=400)[0]["generated_text"].strip()

    monitor.end = time.time()
    answer += f"\n\n---\n⏱ Время: {monitor.end-monitor.start:.2f} с, LLM-запросы: {monitor.llm_calls}, Wiki-запросы: {monitor.wiki_calls}"
    return answer

# ------------------------------------------------------------------ #
# 4. Gradio-интерфейс
# ------------------------------------------------------------------ #
demo = gr.Interface(
    fn=ask_agent,
    inputs=gr.Textbox(label="Введите фразу с японским термином",
                      placeholder="例: В тексте встретилось 江戸幕府. Как перевести?"),
    outputs=gr.Markdown(label="Справка для переводчика"),
    title="JP-RU Historical Terms Helper",
    description="Интеллектуальный глоссарий для япониста. Агент найдёт термин, вытащит русскую викистатью и подскажет перевод.",
    examples=[
        ["В тексте встретилось 江戸幕府. Как правильно перевести?"],
        ["В тексте встретилось 長州藩. Как правильно перевести?"],
        ["В тексте встретилось 幕府海軍. Как правильно перевести?"],
    ],
    cache_examples=False,
)

# ------------------------------------------------------------------ #
# 5. Запуск
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
