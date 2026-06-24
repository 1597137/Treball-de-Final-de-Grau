# Tomografia computada: reconstrucció i regularització

Codi de la part computacional del Treball de Final de Grau sobre reconstrucció d'imatges en tomografia computada.

El projecte genera diferents phantoms, simula sinogrames en diversos casos experimentals i compara tres mètodes de reconstrucció: FBP, Tikhonov i TV.

## Fitxers

- `phantoms.py`: generació dels phantoms.
- `experiments.py`: definició dels casos experimentals.
- `reconstruccio.py`: mètodes de reconstrucció.
- `utilitats.py`: funcions auxiliars, mètriques i visualització.
- `totjunt.py`: execució dels experiments i exportació de resultats.

## Execució

El fitxer principal és:

```bash
python totjunt.py
