"""
Модуль для создания Gradio-интерфейса для JP-RU Historical Terms Helper.

Этот модуль загружает предобученную модель для генерации текста,
определяет функцию для обработки пользовательских запросов и запускает
Gradio-приложение для предоставления веб-интерфейса.
"""

import time
import re
import json
import wikipedia
import requests
from typing import Dict
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# --- 1. Загрузка модели и настройка ---

# Идентификатор модели
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)

# Создание пайплайна для генерации текста
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.05,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)
llm = HuggingFacePipeline(pipeline=pipe)

print(f"Модель {MODEL_ID} загружена.")

# --- 2. Утилиты для работы с Википедией ---

# Заголовки для запросов к API Википедии
HEADERS = {"User-Agent": "ja2ru-helper/0.1 (https://github.com/jhgudleik/)"}

def get_qid_sitelinks(page_title: str, lang: str = "ja") -> Dict[str, str]:
    """
    Получает QID и ссылки на другие языковые разделы Википедии для заданной статьи.

    Args:
        page_title: Название статьи на языке 'lang'.
        lang: Язык, на котором ищется статья (по умолчанию 'ja').

    Returns:
        Словарь, где ключ - код языка, значение - название статьи на этом языке.
        Возвращает пустой словарь, если статья не найдена или не связана с Викиданными.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    wiki_api = f"https://{lang}.wikipedia.org/w/api.php"
    error_message = lambda title, e: print(f"⚠️ Ошибка получения QID для «{title}»: {e}")

    try:
        r = session.get(
            wiki_api,
            params={
                "action": "query",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "titles": page_title,
                "format": "json",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        error_message(page_title, e)
        return {}

    pages = data.get("query", {}).get("pages", {})
    qid = next(iter(pages.values()), {}).get("pageprops", {}).get("wikibase_item")
    if not qid:
        return {}

    try:
        r2 = session.get(
            "https://www.wikidata.org/w/api.php",
            params={"action": "wbgetentities", "ids": qid, "props": "sitelinks", "format": "json"},
            timeout=10,
        )
        r2.raise_for_status()
        data2 = r2.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"⚠️ Ошибка получения sitelinks для {qid}: {e}")
        return {}

    sitelinks = data2.get("entities", {}).get(qid, {}).get("sitelinks", {})
    return {s.replace("wiki", ""): data["title"] for s, data in sitelinks.items() if s.endswith("wiki")}


def ja2ru_wiki_page(ja_term: str):
    """
    Ищет японскую статью по термину, находит соответствующую русскую статью
    и возвращает её заголовок и начало текста.

    Args:
        ja_term: Японский термин для поиска.

    Returns:
        Кортеж (русский_заголовок, текст_статьи). Возвращает (None, None), если
        статья не найдена.
    """
    print(f"Поиск статьи для '{ja_term}'...")
    try:
        wikipedia.set_lang("ja")
        ja_page = wikipedia.page(ja_term, auto_suggest=False)
    except Exception as e:
        print(f"❌ Японская страница не найдена: {e}")
        return None, None

    sitelinks = get_qid_sitelinks(ja_page.title, lang="ja")
    ru_title = sitelinks.get("ru")
    if not ru_title:
        print("❌ Русская интер-вики ссылка не найдена")
        return None, None

    try:
        wikipedia.set_lang("ru")
        ru_page = wikipedia.page(ru_title, auto_suggest=False)
        print(f"Найдена русская статья: '{ru_page.title}'")
        return ru_page.title, ru_page.content[:1500]
    except Exception as e:
        print(f"❌ Не удалось загрузить русскую статью '{ru_title}': {e}")
        return None, None


# --- 3. Основная функция обработки запроса ---

def translate_ja_term_gradio(ja_query: str) -> str:
    """
    Обрабатывает пользовательский запрос, извлекает японский термин,
    находит русскую статью в Википедии и генерирует справку для переводчика.

    Args:
        ja_query: Фраза пользователя, содержащая японский термин.

    Returns:
        Строка со справкой для переводчика, включая метрики производительности.
    """
    start_time = time.time()
    llm_calls = 0
    wiki_calls = 0

    # 1. Извлечение японского термина
    extract_prompt = f"""<|im_start|>system
Ты извлекаешь японский термин из текста пользователя.
1. Ответь **точно теми же иероглифами / каной**, что есть во входе.
2. Не переводи, не транскрибируй, не добавляй пояснений.
3. Выведи **одну непрерывную строку** только из японских символов (кандзи, хирагана, катакана).
4. Если таких символов несколько — верни самую длинную непрерывную последовательность.

Примеры:
Вход: 船手頭是什么？ → 船手頭
Вход: あの人は戦国時代が好きだ → 戦国時代
<|im_start|>user
{ja_query}
<|im_start|>assistant"""

    term_ja = pipe(extract_prompt)[0]["generated_text"].strip()
    print(f"Извлечённый термин: {term_ja}")
    llm_calls += 1

    if not term_ja:
        return "Не удалось извлечь японский термин. Попробуйте другую фразу."

    # 2. Поиск русской статьи в Википедии
    ru_title, ru_text = ja2ru_wiki_page(term_ja)
    wiki_calls += 1 # Увеличиваем счетчик вызовов wiki

    if not ru_title:
        return "Русская статья не найдена. Попробуйте переформулировать запрос или проверить термин."

    # 3. Генерация финального ответа
    final_prompt = f"""<|im_start|>system
Ты — помощник переводчика с японского на русский. Используя факты из русской Википедии (ниже), дай:
1) сообщи, что самый удачный вариант перевода на русский это - заголовок аналогичной статьи в русской Википедии: {ru_title};
2) дай дословный перевод японского термина {term_ja} на русский;
3) дай краткую справку (1-2 предложения) о том, что это такое, используя факты из статьи.

Факты из Википедии:
{ru_text[:1200]}
<|im_start|>user
Как перевести «{term_ja}»?<|im_start|>assistant"""

    answer = pipe(final_prompt)[0]["generated_text"].strip()
    llm_calls += 1

    end_time = time.time()
    total_time = end_time - start_time

    # Добавление метрик к ответу
    answer += f"\n\n---\nВремя: {total_time:.2f} сек, LLM-запросы: {llm_calls}, Wiki-запросы: {wiki_calls}"
    return answer


# --- 4. Настройка и запуск Gradio-интерфейса ---

def translate_ja_term_gradio_wrapper(ja_query: str) -> str:
    """
    Обертка для Gradio, которая вызывает основную функцию и возвращает
    только текстовый результат.
    """
    # Здесь можно добавить логирование или другие подготовительные действия
    return translate_ja_term_gradio(ja_query)


# Создание Gradio-интерфейса
demo = gr.Interface(
    fn=translate_ja_term_gradio_wrapper,
    inputs=gr.Textbox(
        label="Введите фразу с японским термином",
        placeholder="例: В тексте встретилось 江戸幕府. Как перевести?"
    ),
    outputs=gr.Markdown(label="🤖 Справка для переводчика"),
    title="JP-RU Historical Terms Helper",
    description="Интеллектуальный помощник для историков и японистов. Введите предложение "
                "с японским термином, и получите подсказку по его переводу и контексту.",
    examples=[
        ["В тексте встретилось 江戸幕府. Как правильно перевести?"],
        ["В тексте встретилось 長州藩. Как правильно перевести?"],
        ["В тексте встретилось 幕府海軍. Как правильно перевести?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    print("Запуск Gradio-интерфейса...")
    demo.launch()
