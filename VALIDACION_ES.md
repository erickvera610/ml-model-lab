# Evidencia de validación automatizada

Se ejecutaron diez escenarios diferentes con datos ficticios o sintéticos. Este informe describe ejecuciones reales del código mediante un script automatizado. No representa tráfico de usuarios, experiencia laboral ni contribuciones realizadas manualmente por Erik.

Resultado: **10 de 10 escenarios aprobados**.

Consulta `validation-results.json` para ver cada entrada, resultado y fecha UTC de ejecución.

## Repetir las ejecuciones

Desde la carpeta del repositorio ejecuta:

```bash
python validate_scenarios.py
```

En Mercado Aprende instala primero las dependencias del README. En los otros cinco proyectos basta Python. El script utiliza almacenamiento temporal: no cambia registros personales ni realiza compras reales. Regenera el informe local; para conservar nuevos resultados en GitHub debes revisarlos y hacer un commit.

## Qué se comprobó

| Escenario | Entrada | Resultado |
|---|---|---|
| 1 | {"threshold": 0.05} | Aprobado |
| 2 | {"threshold": 0.15} | Aprobado |
| 3 | {"threshold": 0.25} | Aprobado |
| 4 | {"threshold": 0.35} | Aprobado |
| 5 | {"threshold": 0.45} | Aprobado |
| 6 | {"threshold": 0.55} | Aprobado |
| 7 | {"threshold": 0.65} | Aprobado |
| 8 | {"threshold": 0.75} | Aprobado |
| 9 | {"threshold": 0.85} | Aprobado |
| 10 | {"threshold": 0.95} | Aprobado |
