import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st

# OpenAI Python SDK (v1+)
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError


# =========================
# Konfiguracja i stałe
# =========================

DEFAULT_MODEL = "gpt-4.1-mini"  # możesz zmienić na inny model dostępny na Twoim koncie
MAX_RETRIES = 4

SYSTEM_PROMPT = """Jesteś redaktorem medycznym i ekspertem SEO. Twoim zadaniem jest optymalizacja treści medycznej w Markdown pod SEO.
KRYTYCZNE ZASADY BEZPIECZEŃSTWA I RZETELNOŚCI:
- Opieraj się WYŁĄCZNIE na informacjach zawartych w dostarczonym tekście. Nie dodawaj nowych faktów medycznych.
- Nie wymyślaj danych, statystyk, zaleceń, przeciwwskazań ani informacji klinicznych, których nie ma w tekście.
- Jeśli fraza nie pasuje merytorycznie do treści, pomiń ją.
- Nie zmieniaj znaczenia ani wniosków medycznych. Nie dodawaj nowych rekomendacji.
ZASADY EDYCJI:
- Zachowaj format Markdown oraz istniejącą strukturę nagłówków (H2, H3 itd.) i kolejność sekcji.
- Frazy kluczowe wplataj naturalnie (bez keyword stuffingu), głównie w istniejące akapity.
- Nowe sekcje H2 możesz dodawać WYŁĄCZNIE NA KOŃCU dokumentu (po całej istniejącej treści) i tylko, jeśli to konieczne, aby sensownie uwzględnić część fraz.
- Nie dodawaj nowych H2 w środku dokumentu ani nie przemieszczaj istniejących sekcji.
FORMAT ODPOWIEDZI:
- Zwróć WYŁĄCZNIE poprawny JSON w formacie opisanym w wiadomości użytkownika. Bez dodatkowego tekstu.
"""

USER_PROMPT_TEMPLATE = """Masz:
1) Listę fraz kluczowych (każda w osobnej linii)
2) Artykuł medyczny w Markdown

Twoje zadanie:
- Najpierw spróbuj wpleść jak najwięcej fraz w istniejące akapity, bez zmiany faktów medycznych.
- Jeśli po naturalnym wpleceniu zostają frazy, które DA SIĘ sensownie dodać bez nowych faktów medycznych, dodaj na końcu dokumentu jedną lub kilka nowych sekcji H2 z krótkimi akapitami (wciąż bez dodawania faktów).
- Jeśli fraza nie pasuje merytorycznie do treści lub wymagałaby dodania nowych informacji, POMIŃ ją.

Instrukcja obowiązkowa (dosłownie):
"Opieraj się wyłącznie na informacjach zawartych w dostarczonym tekście. Nie dodawaj nowych faktów medycznych. Jeśli fraza nie pasuje merytorycznie do treści, pomiń ją."

Zwróć WYŁĄCZNIE JSON o polach:
- optimized_markdown: string (wynikowy Markdown)
- used_phrases: array[string] (frazy faktycznie użyte w tekście, w brzmieniu dokładnie jak wejście)
- skipped_phrases: array[string] (frazy pominięte)
- brief_notes: array[string] (krótkie notatki np. dlaczego coś pominięto; bez dodawania wiedzy medycznej)

FRAZY (jedna na linię):
{phrases}

TEKST MARKDOWN:
{markdown}
"""


# =========================
# Narzędzia: parsing / walidacje
# =========================

def normalize_phrase_lines(raw: str) -> List[str]:
    """Czyści wejściowe frazy: usuwa puste linie, trimuje, deduplikuje z zachowaniem kolejności."""
    lines = [ln.strip() for ln in (raw or "").splitlines()]
    lines = [ln for ln in lines if ln]
    seen = set()
    out = []
    for ln in lines:
        key = ln.lower()
        if key not in seen:
            out.append(ln)
            seen.add(key)
    return out


def markdown_code_fences_balanced(md: str) -> bool:
    """Sprawdza czy liczba ``` jest parzysta (prosty test poprawności bloków kodu)."""
    return (md.count("```") % 2) == 0


def extract_h2_headings(md: str) -> List[str]:
    """Wyciąga listę nagłówków H2 (## ...)."""
    headings = []
    for m in re.finditer(r"(?m)^\s*##\s+(.+?)\s*$", md):
        headings.append(m.group(1).strip())
    return headings


def validate_h2_added_only_at_end(original_md: str, optimized_md: str) -> Tuple[bool, str]:
    """
    Walidacja: wszystkie oryginalne H2 muszą pojawić się w tej samej kolejności,
    a ewentualne nowe H2 mogą pojawić się dopiero po ostatnim oryginalnym H2 w output.
    """
    orig_h2 = extract_h2_headings(original_md)
    opt_h2 = extract_h2_headings(optimized_md)

    # Jeśli nie było H2 w oryginale, dopuszczamy H2, ale wymagamy, żeby treść oryginalna nie była "pocięta".
    # (W praktyce: zostawiamy tylko miękką walidację.)
    if not orig_h2:
        return True, "Brak H2 w oryginale — pomijam twardą walidację kolejności H2."

    # Sprawdź, czy orig_h2 jest podciągiem opt_h2 w tej samej kolejności.
    i = 0
    for h in opt_h2:
        if i < len(orig_h2) and h == orig_h2[i]:
            i += 1
    if i != len(orig_h2):
        return False, "Wynik narusza kolejność/obecność istniejących nagłówków H2 (zostały zmienione, usunięte lub przestawione)."

    # Znajdź indeks ostatniego oryginalnego H2 w opt_h2
    last_idx = -1
    for idx, h in enumerate(opt_h2):
        if h == orig_h2[-1]:
            last_idx = idx

    # Jeśli w opt_h2 są dodatkowe H2 przed last_idx, to znaczy, że nowe H2 wstawiono w środku.
    # Ale uwaga: opt_h2 może zawierać te same nazwy H2 (duplikaty). Wymagamy tylko, aby
    # pierwsze wystąpienie sekwencji oryginalnych H2 było zachowane i nowe H2 nie wchodziły pomiędzy.
    # Prostsze: sprawdzamy, czy wszystkie H2 przed "ukończeniem" sekwencji oryginalnej są równe odpowiednim oryginałom.
    seq_pos = 0
    for h in opt_h2:
        if seq_pos < len(orig_h2) and h == orig_h2[seq_pos]:
            seq_pos += 1
        else:
            # jeśli jeszcze nie domknęliśmy wszystkich oryginalnych H2, a trafiamy na inny H2 -> nowe H2 w środku
            if seq_pos < len(orig_h2):
                return False, "Wynik wstawia nowe sekcje H2 w środku dokumentu. Nowe H2 mogą być tylko na końcu."

    return True, "Walidacja H2 OK (nowe H2 — jeśli są — znajdują się po całej istniejącej strukturze)."


def count_phrase_usage_by_presence(phrases: List[str], optimized_md: str) -> Dict[str, bool]:
    """
    Liczy użycie fraz przez sprawdzenie obecności (case-insensitive) w tekście wynikowym.
    Uwaga: to nie jest idealne (odmiany fleksyjne), ale jest proste i przewidywalne.
    """
    text = optimized_md.lower()
    usage = {}
    for ph in phrases:
        usage[ph] = ph.lower() in text
    return usage


# =========================
# OpenAI: wywołanie + retry + JSON parse
# =========================

@dataclass
class OptimizeResult:
    optimized_markdown: str
    used_phrases: List[str]
    skipped_phrases: List[str]
    brief_notes: List[str]
    raw_json: Dict


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY w zmiennych środowiskowych.")
    return OpenAI(api_key=api_key)


def call_openai_with_retries(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            last_err = e
            # prosty backoff
            sleep_s = min(2 ** attempt, 16)
            time.sleep(sleep_s)

    raise RuntimeError(f"Nie udało się wykonać zapytania do API po {MAX_RETRIES} próbach. Ostatni błąd: {last_err}")


def parse_strict_json(text: str) -> Dict:
    """
    Model ma zwrócić wyłącznie JSON, ale w praktyce bywa różnie.
    Ta funkcja próbuje:
    - wczytać całość jako JSON,
    - jeśli nie wyjdzie, wyciąć największy blok { ... } i wczytać ponownie.
    """
    text = (text or "").strip()

    # 1) próba wprost
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) wytnij największy blok JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        return json.loads(chunk)

    raise ValueError("Nie udało się sparsować JSON z odpowiedzi modelu.")


def optimize_markdown_medical(
    phrases: List[str],
    markdown: str,
    model: str,
    temperature: float,
) -> OptimizeResult:
    client = get_openai_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        phrases="\n".join(phrases),
        markdown=markdown,
    )

    raw = call_openai_with_retries(
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
    )

    data = parse_strict_json(raw)

    optimized = str(data.get("optimized_markdown", "")).strip()
    used = data.get("used_phrases", []) or []
    skipped = data.get("skipped_phrases", []) or []
    notes = data.get("brief_notes", []) or []

    # Minimalne sanity-checki typów
    if not isinstance(used, list):
        used = []
    if not isinstance(skipped, list):
        skipped = []
    if not isinstance(notes, list):
        notes = []

    return OptimizeResult(
        optimized_markdown=optimized,
        used_phrases=[str(x) for x in used],
        skipped_phrases=[str(x) for x in skipped],
        brief_notes=[str(x) for x in notes],
        raw_json=data,
    )


# =========================
# Streamlit UI
# =========================

def clipboard_button(markdown_text: str):
    """
    Przycisk kopiowania do schowka z użyciem małego kawałka JS.
    Działa w większości przeglądarek; w niektórych środowiskach może wymagać https.
    """
    import streamlit.components.v1 as components

    escaped = markdown_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <button style="padding:0.5rem 0.75rem;border-radius:0.5rem;border:1px solid #ddd;cursor:pointer;"
      onclick="navigator.clipboard.writeText(`{escaped}`)">
      📋 Kopiuj wynik do schowka
    </button>
    """
    components.html(html, height=55)


def main():
    st.set_page_config(page_title="Surfer SEO – Medical Markdown Optimizer", layout="wide")

    st.title("Surfer SEO – Medical Markdown Optimizer (Markdown → Markdown)")
    st.caption(
        "Wplata frazy kluczowe naturalnie w istniejący tekst medyczny bez dodawania nowych faktów. "
        "Nowe sekcje H2 — tylko na końcu (jeśli konieczne)."
    )

    with st.sidebar:
        st.header("Ustawienia")
        model = st.text_input("Model OpenAI", value=DEFAULT_MODEL, help="Np. gpt-4.1-mini / inny dostępny model")
        temperature = st.slider("Temperature", 0.0, 0.8, 0.2, 0.05, help="Niżej = bardziej zachowawczo i spójnie")
        st.markdown("---")
        st.markdown("**Wymagane:** ustaw `OPENAI_API_KEY` w zmiennych środowiskowych.")
        st.markdown("**Bezpieczeństwo medyczne:** narzędzie nie może dodawać nowych faktów — tylko redaguje i porządkuje.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("1) Frazy z Surfer SEO (1 fraza na linię)")
        phrases_raw = st.text_area(
            "Frazy",
            height=260,
            placeholder="np.\nobjawy alergii\nleczenie alergii\n...",
            label_visibility="collapsed",
        )
        phrases = normalize_phrase_lines(phrases_raw)
        st.write(f"Frazy po czyszczeniu: **{len(phrases)}**")

    with col2:
        st.subheader("2) Tekst wejściowy (Markdown)")
        markdown_in = st.text_area(
            "Markdown wejściowy",
            height=260,
            placeholder="Wklej artykuł w Markdown...",
            label_visibility="collapsed",
        )
        st.write(f"Długość tekstu: **{len(markdown_in)}** znaków")

    st.markdown("---")

    run = st.button("🚀 Optymalizuj", type="primary", use_container_width=True, disabled=(not phrases or not markdown_in))

    if run:
        # Szybkie walidacje wejścia
        if not markdown_code_fences_balanced(markdown_in):
            st.error("Wejściowy Markdown ma nieparzystą liczbę znaczników ``` (prawdopodobnie niezamknięty blok kodu). Popraw i spróbuj ponownie.")
            st.stop()

        try:
            with st.status("Przetwarzam treść i wplatam frazy...", expanded=True) as status:
                status.write("Łączenie z API OpenAI…")
                result = optimize_markdown_medical(
                    phrases=phrases,
                    markdown=markdown_in,
                    model=model,
                    temperature=temperature,
                )

                status.write("Walidacja struktury Markdown…")
                if not result.optimized_markdown:
                    raise RuntimeError("Model zwrócił pusty wynik. Spróbuj ponownie albo zmień model/temperature.")

                if not markdown_code_fences_balanced(result.optimized_markdown):
                    st.warning("Uwaga: wynikowy Markdown ma nieparzystą liczbę znaczników ``` (może wymagać ręcznej korekty).")

                ok_h2, msg_h2 = validate_h2_added_only_at_end(markdown_in, result.optimized_markdown)
                if not ok_h2:
                    st.error(msg_h2)
                    st.info("Wskazówka: zmniejsz temperature i spróbuj ponownie albo dopisz w tekście więcej kontekstu do fraz.")
                    st.stop()
                else:
                    status.write(msg_h2)

                status.update(label="Gotowe ✅", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Błąd: {e}")
            st.stop()

        # Liczenie użycia: (a) deklaratywnie z JSON (b) weryfikacja presence
        declared_used = set([p.strip() for p in result.used_phrases])
        declared_skipped = set([p.strip() for p in result.skipped_phrases])

        presence = count_phrase_usage_by_presence(phrases, result.optimized_markdown)
        present_used = {p for p, is_used in presence.items() if is_used}
        present_unused = {p for p, is_used in presence.items() if not is_used}

        st.subheader("3) Wynik (Markdown)")
        st.text_area("Zoptymalizowany Markdown", value=result.optimized_markdown, height=380)

        # Akcje: kopiuj + pobierz
        a1, a2 = st.columns([1, 1], gap="large")
        with a1:
            clipboard_button(result.optimized_markdown)
        with a2:
            st.download_button(
                "💾 Pobierz jako .md",
                data=result.optimized_markdown.encode("utf-8"),
                file_name="optimized.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("4) Licznik fraz i raport")

        # Panel metryk
        m1, m2, m3 = st.columns(3)
        m1.metric("Frazy wejściowe", len(phrases))
        m2.metric("Użyte (wykryte w tekście)", len(present_used))
        m3.metric("Nieużyte (wykryte)", len(present_unused))

        with st.expander("Szczegóły: użyte / nieużyte"):
            st.markdown("**Użyte (wykryte w wynikowym Markdown):**")
            st.write(sorted(present_used))
            st.markdown("**Nie użyte (wykryte):**")
            st.write(sorted(present_unused))

        with st.expander("Szczegóły: deklaracja modelu (used_phrases / skipped_phrases)"):
            st.markdown("**used_phrases (deklaracja):**")
            st.write(sorted(declared_used))
            st.markdown("**skipped_phrases (deklaracja):**")
            st.write(sorted(declared_skipped))

        if result.brief_notes:
            with st.expander("Notatki modelu (brief_notes)"):
                for n in result.brief_notes:
                    st.write(f"- {n}")

        with st.expander("Debug: surowy JSON z modelu"):
            st.json(result.raw_json)


if __name__ == "__main__":
    main()
