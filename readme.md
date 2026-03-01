# Surfer SEO – Medical Markdown Optimizer

Aplikacja Streamlit do optymalizacji istniejących artykułów medycznych w Markdown pod frazy z Surfer SEO:
- wplata frazy naturalnie w istniejące akapity (bez keyword stuffingu),
- NIE dodaje nowych faktów medycznych (opiera się tylko na tekście wejściowym),
- nowe sekcje H2 (jeśli konieczne) dodaje WYŁĄCZNIE na końcu dokumentu,
- pokazuje licznik użytych fraz,
- pozwala skopiować wynik do schowka i pobrać `.md`.

## Wymagania
- Python 3.10+
- Klucz API: `OPENAI_API_KEY`

## Instalacja
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
