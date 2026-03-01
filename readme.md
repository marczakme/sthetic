# Surfer SEO – Medical Optimizer (Markdown in → DOCX out)

Agresywniejsza optymalizacja pod Surfer SEO:
- używa fraz literalnie (dokładne brzmienie),
- rozkłada frazy w istniejących akapitach,
- jeśli trzeba, dodaje NA KOŃCU dokumentu sekcje H2: **Podsumowanie** i **FAQ**,
- treść nowych sekcji to wyłącznie powtórzenie/parafraza informacji z artykułu (zero nowych faktów medycznych),
- eksportuje wynik do **DOCX**.

## Instalacja
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
