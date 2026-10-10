# Tutorial en español: Classification Evaluation Lab

## 1. Qué problema resuelve

Una probabilidad y una decisión no son lo mismo. El modelo produce probabilidades; un umbral decide la clase. Evaluamos ambos aspectos por separado.

## 2. Pruébalo primero

Abre https://erickvera610.github.io/ml-model-lab/ . La demo funciona en tu navegador. Cambia un control y compara la salida. Los datos ficticios no son usuarios reales; los ejemplos tampoco representan trabajo realizado para una empresa.

## 3. Instálalo paso a paso

1. Instala Python desde python.org y Git desde git-scm.com.
2. Abre PowerShell y escribe `python --version` y `git --version`.
3. Escribe `git clone https://github.com/erickvera610/ml-model-lab.git`.
4. Escribe `cd ml-model-lab`.
5. Escribe `python app.py` y mantén abierta la terminal.
6. Abre http://127.0.0.1:8000 y prueba los controles.
7. Detén el servidor con Ctrl+C.

## 4. La secuencia de trabajo

`300 training observations → 100 validation observations → selected threshold → 100 test observations`

**Sigmoid** transforma una suma ponderada en probabilidad. **Log-loss** penaliza probabilidades equivocadas. **Gradiente descendente** cambia los pesos para bajar esa pérdida. **Precision** pregunta cuántos positivos predichos fueron correctos; **recall** pregunta cuántos positivos reales encontramos. **F1** combina ambos. **AUC** mide ordenamiento: no depende de un solo umbral. **Brier score** es el promedio de (probabilidad − etiqueta)²; menor suele ser mejor. **Calibración** compara probabilidades con frecuencias observadas.

El umbral se escoge con los 100 casos de validación. Los otros 100 casos de prueba no participan en esa elección. Modificar el umbral mirando repetidamente la prueba convertiría esa prueba en otra validación.

## 5. Leer el código con calma

Empieza por `core.py` y busca `run`. Después abre el archivo especializado indicado en README. Sigue una sola entrada hasta su salida. Lee las pruebas: cada una describe un comportamiento concreto. Por último, revisa `enhanced.js` para comparar la versión del navegador.

## 6. Reproducir los resultados

```bash
python -m unittest discover -v
node test_enhanced.cjs
python benchmark.py
```

Abre `benchmark-results.json` y compara su resumen con `CASE_STUDY.md`. El reporte incluye una hora real de ejecución. Los experimentos con semillas repetidas muestran varios resultados; no escogemos solo el mejor. Las métricas sintéticas de JavaScript pueden diferir de Python.

## 7. Tu siguiente mejora personal

Implementa un costo distinto para falsos positivos y falsos negativos. Selecciona el umbral en validación según ese costo y compara con F1, manteniendo intacta la prueba final.

```bash
git switch -c feature/my-improvement
# Edita código y ejecuta las pruebas
git add .
git commit -m "Describe the actual behavior change"
git push -u origin feature/my-improvement
```

Abre un pull request y explica problema, solución, resultado y límite. Se usó asistencia de IA para construir esta base. En una entrevista distingue lo que recibiste, lo que estudiaste y lo que realmente cambiaste.

## 8. Practicar la entrevista

- What problem does this application solve?
- Which data are available when the prediction is made?
- What is the simplest baseline?
- How did you avoid data leakage or inconsistent state?
- Which result surprised you?
- What failure case remains?
- What did you personally improve?

Primero responde en español. Después prepara cuatro frases cortas en inglés: problema, método, resultado y limitación. Usa solo resultados documentados.
