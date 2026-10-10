# Cómo entender y presentar este proyecto

1. Lee primero el problema y los datos en README.md.
2. Ejecuta la demo para entender las entradas y salidas.
3. Ejecuta las pruebas: `python -m unittest discover -v`.
4. Reproduce el experimento indicado abajo. Mira el resultado y compáralo con la referencia sencilla.
5. Explica qué datos se reservaron para prueba, qué decisiones se tomaron antes y en qué casos falla.

Instala `requirements-benchmark.txt` antes de las pruebas. Ejecuta `python public_benchmark.py`. El modelo reconoce dígitos reales de 8 × 8 píxeles. La validación escoge entre dos modelos. La prueba final mide el ganador con imágenes que no utilizó para aprender. 96,67% corresponde a esta partición concreta, no a toda imagen posible.

Para una entrevista, prepara una explicación de dos minutos: problema → datos → método → comparación → resultado → límites. Identifica qué código te ayudó a generar IA y demuestra que entiendes cada decisión.
