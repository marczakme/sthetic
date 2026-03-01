"""
Streamlit app: Surfer-oriented FAQ builder

- Wejścia:
  1) frazy (1 na linię)
  2) oryginalny tekst (Markdown)
  3) fakty AI (1 fakt na linię) — dopuszczone dodatkowe źródło faktów

- Logika:
  * Model otrzymuje oryginalny tekst + fakty AI + listę fraz.
  * Generuje strukturę FAQ (dla każdej pasującej frazy tworzy jedną sekcję H2 z krótkim akapitem).
  * FAQ ma unikać powielania informacji (każdy paragraf wnosi unikalny kontekst).
  * MODEL MOŻE KORZYSTAĆ WYŁĄCZNIE Z TEKSTU WEJŚCIOWEGO I Z FAKTÓW AI — NIE DOPISUJE NOWYCH FAKTÓW MEDYCZNYCH.
  * Jeśli fraza nie jest merytorycznie dopasowana, model ma ją pominąć.
  * Output: JSON zawierający 'faq_markdown' (cały markdown FAQ), 'used_phrases', 'skipped_phrases', 'brief_notes'.

- Wynik: wyświetlany w aplikacji pole z Markdown oraz przycisk kopiowania.
"""

import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

import streamlit as st
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

# ==========================
# Ustawienia domyślne
# ==========================
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3

# System prompt — twarde zasady rzetelności medycznej
SYSTEM_PROMPT = """Jesteś redaktorem medycznym i specjalistą SEO. Masz ścisły zakaz dodawania jakichkolwiek nowych faktów medycznych poza tym, co dostarczono.
Korzystaj WYŁĄCZNIE z:

1) tekstu wejściowego (Markdown) podanego przez użytkownika,
2) pól 'Fakty AI' dostarczonych przez użytkownika.

NIE WYMYSZAJ nowych faktów, zaleceń ani statystyk. Jeśli fraza nie pasuje merytorycznie do tekstu i faktów AI — pomiń ją.
Twój cel: zbudować strukturę FAQ (sekcja: H2 + jeden krótki paragraf) wykorzystując frazy i fakty, tak aby FAQ:
- nie powielał informacji (unikalność paragrafów),
- naturalnie wplatał frazy w pytania lub odpowiedzi tam, gdzie pasują,
- był gotowy do skopiowania jako Markdown (nagłówki H2: '## ...' oraz akapity).
Zwróć WYŁĄCZNIE JSON bez dodatkowego tekstu, o polach:
- faq_markdown: string (cały wynikowy markdown z FAQ)
- used_phrases: array[string]
- skipped_phrases: array[string]
- brief_notes: array[string] (krótkie notatki: dlaczego fraza została pominięta lub jak użyto faktów)
"""

# Template user prompt — zostanie wypełniony danymi
USER_PROMPT_TEMPLATE = """
Masz:
- listę fraz (jedna fraza na linię):
{phrases}

- tekst źródłowy (Markdown):
{markdown}

- fakty AI (jedna po linii), które możesz wykorzystać jako dodatkowe dozwolone źródło:
{facts_ai}

Zadanie:
1) Dla każdej frazy, która jest merytorycznie dopasowana do tekstu lub faktów AI, stwórz jedną sekcję FAQ:
   - linia H2: '## Pytanie' (forma pytania powinna naturalnie zawierać frazę, jeśli to możliwe)
   - krótki paragraf (1-3 zdania) jako odpowiedź, korzystający tylko z treści tekstu źródłowego i/lub faktów AI.
2) Jeśli fraza NIE pasuje merytorycznie, pomiń ją i dodaj krótki wpis w 'brief_notes' dlaczego.
3) Unikaj powielania tych samych informacji — upewnij się, że każdy FAQ wnosi unikalny punkt widzenia/fragment informacji.
4) NIE DODAWAJ nowych faktów medycznych poza dostarczonymi.
5) Odpowiedzi mają być zwięzłe, rzeczowe, w tonie profesjonalnym.

Zwróć WYŁĄCZNIE JSON z polami:
- faq_markdown (string),
- used_phrases (array),
- skipped_phrases (array),
- brief_notes (array).
"""

# ==========================
# Typy wyników pomocnicze
# ==========================
@dataclass
class OptimizeResult:
    faq_markdown: str
    used_phrases: List[str]
    skipped_phrases: List[str]
    brief_notes: List[str]
    raw_text: str

# ==========================
# Funkcje pomocnicze
# ==========================
def normalize_lines(raw: str) -> List[str]:
    """Usuń puste, przytnij, deduplikuj zachowując kolejność."""
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    out = []
    seen = set()
    for ln in lines:
        if not ln:
            continue
        key = ln.lower()
        if key not in seen:
            out.append(ln)
            seen.add(key)
    return out

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych.")
    return OpenAI(api_key=api_key)

def call_openai_with_retries(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.2) -> str:
    client = get_openai_client()
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Błąd OpenAI po {MAX_RETRIES} próbach: {last_err}")

def extract_json_from_response(text: str) -> Dict:
    """Próba wydobycia JSONa z odpowiedzi modelu (najpierw cały tekst -> json, potem szukanie bloku)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        # spróbuj wyciągnąć fragment od pierwszego { do ostatniego }
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            chunk = text[s:e+1]
            return json.loads(chunk)
    raise ValueError("Nie udało się sparsować JSON z odpowiedzi modelu.")

# ==========================
# Streamlit UI
# ==========================
def clipboard_button(markdown_text: str):
    """Mały JS do kopiowania do schowka."""
    import streamlit.components.v1 as components
    escaped = markdown_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <button style="padding:0.5rem 0.75rem;border-radius:0.4rem;border:1px solid #ddd;cursor:pointer;"
      onclick="navigator.clipboard.writeText(`{escaped}`)">
      📋 Kopiuj FAQ do schowka
    </button>
    """
    components.html(html, height=55)

def main():
    st.set_page_config(page_title="Surfer FAQ Builder (Markdown → FAQ)", layout="wide")
    st.title("Surfer FAQ Builder — buduj FAQ z fraz i faktów AI")
    st.caption("Wprowadz frazy, tekst (Markdown) i fakty AI — aplikacja wygeneruje FAQ (H2 + krótki paragraf). Wynik kopiujesz do schowka i wklejasz do Surfera.")

    with st.sidebar:
        st.header("Ustawienia")
        model = st.text_input("Model OpenAI", value=DEFAULT_MODEL)
        temperature = st.slider("Temperature", 0.0, 0.7, 0.2, 0.05)
        st.markdown("---")
        st.markdown("Uwaga: ustaw `OPENAI_API_KEY` w zmiennych środowiskowych przed uruchomieniem.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Frazy (1 fraza na linię)")
        phrases_raw = st.text_area("Frazy", height=220, placeholder="np.\nobjawy migreny\nleczenie migreny\n...", label_visibility="collapsed")
        phrases = normalize_lines(phrases_raw)
        st.write(f"Liczba fraz: **{len(phrases)}**")

    with col2:
        st.subheader("2) Tekst źródłowy (Markdown)")
        markdown_in = st.text_area("Markdown wejściowy", height=220, placeholder="Wklej artykuł w Markdown...", label_visibility="collapsed")
        st.write(f"Długość tekstu: **{len(markdown_in)}** znaków")

    st.subheader("3) Fakty AI (dodatkowe, dozwolone źródła) — 1 fakt na linię")
    facts_raw = st.text_area("Fakty AI", height=140, placeholder="np.\nMigrena często rozpoczyna się w wieku dorosłym...\nCzynniki X są związane z Y...", label_visibility="collapsed")
    facts = normalize_lines(facts_raw)
    st.write(f"Liczba faktów AI: **{len(facts)}**")

    st.markdown("---")
    run = st.button("🧭 Generuj FAQ (H2 + paragrafy)", disabled=(not phrases or not markdown_in))

    if run:
        # Walidacja prostych warunków
        if not phrases:
            st.error("Wprowadź co najmniej jedną frazę.")
            st.stop()
        if not markdown_in.strip():
            st.error("Wprowadź tekst źródłowy w Markdown.")
            st.stop()

        # Przygotuj prompt
        phrases_block = "\n".join(phrases)
        facts_block = "\n".join(facts) if facts else "(brak dodatkowych faktów AI)"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            phrases=phrases_block,
            markdown=markdown_in,
            facts_ai=facts_block
        )

        # Wywołanie OpenAI
        try:
            with st.status("Łączenie z OpenAI i generowanie FAQ…", expanded=True) as status:
                status.write("Wysyłam zapytanie do modelu...")
                raw = call_openai_with_retries(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, model=model, temperature=temperature)
                status.write("Otrzymano odpowiedź. Parsuję JSON...")
                parsed = extract_json_from_response(raw)

                faq_md = parsed.get("faq_markdown", "")
                used = parsed.get("used_phrases", []) or []
                skipped = parsed.get("skipped_phrases", []) or []
                notes = parsed.get("brief_notes", []) or []

                # Sanity checks
                if not isinstance(used, list):
                    used = []
                if not isinstance(skipped, list):
                    skipped = []
                if not isinstance(notes, list):
                    notes = []

                result = OptimizeResult(
                    faq_markdown=str(faq_md),
                    used_phrases=[str(x) for x in used],
                    skipped_phrases=[str(x) for x in skipped],
                    brief_notes=[str(x) for x in notes],
                    raw_text=raw
                )

                status.update(label="Gotowe ✅", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Błąd podczas generowania: {e}")
            st.stop()

        # Wyświetl wynik i przycisk kopiowania
        st.subheader("Wynik: FAQ (Markdown)")
        st.markdown("Skopiuj FAQ i wklej do Surfera (Surfer rozpozna nagłówki H2).")
        st.text_area("FAQ Markdown", value=result.faq_markdown, height=360)

        # Kopiowanie
        clipboard_button(result.faq_markdown)

        # Raport fraz
        st.markdown("---")
        st.subheader("Raport użycia fraz")
        left, right = st.columns(2)
        with left:
            st.markdown("**Użyte frazy (deklaracja modelu):**")
            st.write(result.used_phrases)
        with right:
            st.markdown("**Pominięte frazy (deklaracja modelu):**")
            st.write(result.skipped_phrases)

        if result.brief_notes:
            with st.expander("Notatki modelu (brief_notes)"):
                for n in result.brief_notes:
                    st.write(f"- {n}")

        with st.expander("Surowa odpowiedź modelu (do debugu)"):
            st.code(result.raw_text)

if __name__ == "__main__":
    main()
