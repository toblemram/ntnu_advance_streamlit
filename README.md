# NTNU TBM Advance Rate – Streamlit-prototype

Denne mappen er et enkelt utgangspunkt for en Streamlit-app som bruker Excel-filen
`NTNU_Advance rate_Approach 1_rev1.xlsm` som datakilde.

## Struktur

- `app.py` – hovedfilen for Streamlit-appen.
- `requirements.txt` – Python-avhengigheter.
- `data/` – inneholder Excel-filen.
- `src/`
  - `data_loader.py` – funksjoner for å lese og strukturere data fra Excel.
  - `analysis.py` – korrelasjon og sensitivitet (lineær regresjon).
  - `visuals.py` – Plotly-figurer brukt i appen.

## Kjøre lokalt

```bash
pip install -r requirements.txt
streamlit run app.py
```

Appen bruker Excel-filen statisk (ingen tilbakekobling av endringer til Excel).
Neste steg kan være å koble Streamlit mot selve Excel-modellen via f.eks. `xlwings`
for ekte "what-if"-analyse.
