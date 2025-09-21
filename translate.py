import os
import re
import json
import time
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel
from operator import add
from langgraph.graph import StateGraph, START, END
from bert_score import score as bert_score
from rouge import Rouge
from evaluation import EVALUATION_METRICS, get_evaluation_prompt

load_dotenv()
SEALION_API_KEY = os.getenv("SEALION_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

MODEL = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
LANGUAGES = ["Chinese", "Malay"]
N_SENTENCES = 5
N_VOCABULARY = 25


class Translation(BaseModel):
    language: str
    translation: str
    vocabulary: list[dict]


def merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class State(TypedDict):
    content: str
    summary: str
    scores: Annotated[dict, merge]
    translations: Annotated[list[Translation], add]


def invoke_sealion(prompt: str, delay: int = 12) -> dict:
    if not SEALION_API_KEY:
        raise Exception("SEALION_API_KEY not found")

    start_time = time.time()

    url = "https://api.sea-lion.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SEALION_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 2000,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]

    elapsed_time = time.time() - start_time
    print(f"    Elapsed time: {elapsed_time:.1f}s")
    if elapsed_time < delay:
        sleep_time = delay - elapsed_time
        print(f"    Sleep time: {sleep_time:.1f}s")
        time.sleep(sleep_time)

    return json.loads(content)


def filter_articles(articles: list[dict], hours: int = 1) -> list[dict]:
    print("Filtering articles...")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    recent_articles = []
    for article in articles:
        if "published_at" in article and article["published_at"]:
            try:
                published_time = datetime.fromisoformat(article["published_at"])
                if published_time.tzinfo is None:
                    singapore_tz = timezone(timedelta(hours=8))
                    published_time = published_time.replace(tzinfo=singapore_tz)
                if published_time >= cutoff:
                    recent_articles.append(article)
            except ValueError:
                continue

    print(f"Recent articles: {len(recent_articles)}")

    return recent_articles


def summarise_article(state: State) -> dict:
    print("  Summarising in English...")

    prompt = f"""
Summarise the following news article in {N_SENTENCES} English sentences. The summary should be coherent and grammatically correct (e.g. remove ALL double spaces).

Article: {state["content"]}

Return your response as JSON in this format:

{{"summary": "your summary here"}}
"""

    result = invoke_sealion(prompt)
    summary = result.get("summary", "")

    return {"summary": summary}


def evaluate_summary(state: State) -> dict:
    print("  Evaluating English summary...")

    # BERTScore
    P, R, F1 = bert_score(
        [state["summary"]],
        [state["content"]],
        lang="en",
        model_type="bert-base-uncased",
    )
    bert_p = P.item()
    bert_r = R.item()
    bert_f1 = F1.item()

    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(state["summary"], state["content"])
    rouge_1 = rouge_scores[0]["rouge-1"]["f"]
    rouge_2 = rouge_scores[0]["rouge-2"]["f"]
    rouge_l = rouge_scores[0]["rouge-l"]["f"]

    # G-Eval (LLM evaluation)
    eval_scores = {}
    for eval_type, (criteria, steps) in EVALUATION_METRICS.items():
        print(f"    Evaluating {eval_type}...")

        prompt = get_evaluation_prompt(
            criteria=criteria,
            steps=steps,
            document=state["content"],
            summary=state["summary"],
            metric_name=eval_type,
        )

        result = invoke_sealion(prompt)

        if isinstance(result, dict) and eval_type in result:
            eval_score = result[eval_type]
        else:
            eval_score = 1  # default fallback

        eval_scores[eval_type.lower()] = eval_score

    return {
        "scores": {
            "bert_p": bert_p,
            "bert_r": bert_r,
            "bert_f1": bert_f1,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l,
            "relevance": eval_scores.get("relevance", 1),
            "coherence": eval_scores.get("coherence", 1),
            "consistency": eval_scores.get("consistency", 1),
            "fluency": eval_scores.get("fluency", 1),
        }
    }


def translate_article(lang: str):
    """Translation node factory."""

    def _fn(state: State) -> dict:
        print(f"  Translating to {lang}...")

        translate_prompt = f"""
Translate the following summary SENTENCE-BY-SENTENCE into {lang} and return ONLY the translated summary. The translated summary should be coherent and grammatically correct. Each sentence in the translated summary should be separated by ONLY ONE space.

Summary: {state["summary"]}

Return your response as JSON in this format:

{{"translation": "your {lang} translation here"}}
"""

        result = invoke_sealion(translate_prompt)
        translation = result.get("translation", "")

        print(f"  Extracting {lang} vocabulary...")

        extract_prompt = f"""
Extract EXACTLY {N_VOCABULARY} important vocabulary words from ONLY the following {lang} text. For each extracted word (EXCLUDE 拼音), provide ONLY ONE concise English meaning.

{lang} Text: {translation}

Return your response as JSON in this format:

{{"vocabulary": [{{"word": "word1", "meaning": "meaning1"}}, {{"word": "word2", "meaning": "meaning2"}}]}}
"""

        result = invoke_sealion(extract_prompt)
        vocabulary = result.get("vocabulary", [])

        return {
            "translations": [
                {
                    "language": lang,
                    "translation": translation,
                    "vocabulary": vocabulary,
                }
            ]
        }

    return _fn


def format_output(state: State, article: dict) -> dict:
    translation = {item["language"].lower(): item["translation"] for item in state["translations"]}
    vocabulary = {
        item["language"].lower(): {vocab["word"]: vocab["meaning"] for vocab in item["vocabulary"]}
        for item in state["translations"]
    }

    return {
        "article_url": article.get("article_url", ""),
        "image_url": article.get("image_url", ""),
        "title": article.get("title", ""),
        "category": article.get("category", ""),
        "published_at": article.get("published_at", ""),
        "summary": state["summary"],
        "scores": state["scores"],
        "translation": translation,
        "vocabulary": vocabulary,
    }


def verify_outputs(outputs: list[dict]) -> None:
    for i, article in enumerate(outputs):
        print(f"Verifying article {i + 1}/{len(outputs)}...")

        summary = article["summary"]
        sentence_count = sum(summary.count(punct) for punct in ".!?")

        print(f"  English: {sentence_count} sentences")

        scores = article["scores"]
        bert_p = scores["bert_p"]
        bert_r = scores["bert_r"]
        bert_f1 = scores["bert_f1"]
        rouge_1 = scores["rouge_1"]
        rouge_2 = scores["rouge_2"]
        rouge_l = scores["rouge_l"]
        relevance = scores["relevance"]
        coherence = scores["coherence"]
        consistency = scores["consistency"]
        fluency = scores["fluency"]

        print(f"    BERTScore - P: {bert_p:.3f}, R: {bert_r:.3f}, F1: {bert_f1:.3f}")
        print(f"    ROUGE - 1: {rouge_1:.3f}, 2: {rouge_2:.3f}, L: {rouge_l:.3f}")
        print(
            f"    G-Eval - Relevance: {relevance}, Coherence: {coherence}, Consistency: {consistency}, Fluency: {fluency}"
        )

        for lang, translation in article["translation"].items():
            text = translation.lower()
            vocab_dict = article["vocabulary"][lang]
            if lang == "chinese":
                vocab_found = sum(1 for word in vocab_dict if word.lower() in text)
            else:
                vocab_found = sum(
                    1
                    for word in vocab_dict
                    if re.search(r"\b" + re.escape(word.lower()) + r"\b", text)
                )

            print(f"  {lang.title()}: {vocab_found}/{len(vocab_dict)} vocabulary")


if __name__ == "__main__":
    with open("data.json", "r") as f:
        scraped = json.load(f)
    articles = filter_articles(scraped)

    workflow = StateGraph(State)

    # nodes
    workflow.add_node("summarise", summarise_article)
    workflow.add_node("evaluate", evaluate_summary)
    for lang in LANGUAGES:
        workflow.add_node(f"translate_{lang}", translate_article(lang))

    # edges
    workflow.add_edge(START, "summarise")
    workflow.add_edge("summarise", "evaluate")
    for lang in LANGUAGES:
        workflow.add_edge("evaluate", f"translate_{lang}")
        workflow.add_edge(f"translate_{lang}", END)

    graph = workflow.compile()
    # with open("graph.png", "wb") as f:
    #     f.write(graph.get_graph().draw_mermaid_png())

    outputs = []
    for i, article in enumerate(articles):
        print(f"Processing article {i + 1}/{len(articles)}...")

        state = graph.invoke({"content": article["content"]})
        output = format_output(state, article)
        outputs.append(output)

    verify_outputs(outputs)

    # save locally
    # if outputs:
    #     with open("translations.json", "w", encoding="utf-8") as f:
    #         json.dump(outputs, f, indent=2, ensure_ascii=False)

    # save to Supabase
    if outputs:
        print("Saving translations...")
        try:
            from supabase import create_client, Client

            supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

            # upload translations
            response = supabase.table("translations").upsert(outputs).execute()

            # clean table (retain latest 250 articles)
            supabase.rpc("clean_table").execute()

            print(f"  Upload success: {len(response.data)} translations")
        except Exception as e:
            print(f"  Upload failed: {str(e)}")
