# Cómo entender y presentar este proyecto

1. Lee primero el problema y los datos en README.md.
2. Ejecuta la demo para entender las entradas y salidas.
3. Ejecuta las pruebas: `python -m unittest discover -v`.
4. Reproduce el experimento indicado abajo. Mira el resultado y compáralo con la referencia sencilla.
5. Explica qué datos se reservaron para prueba, qué decisiones se tomaron antes y en qué casos falla.

Instala `requirements-benchmark.txt` antes de las pruebas. Ejecuta `python public_benchmark.py`. El modelo reconoce dígitos reales de 8 × 8 píxeles. La validación escoge entre dos modelos. La prueba final mide el ganador con imágenes que no utilizó para aprender. 96,67% corresponde a esta partición concreta, no a toda imagen posible.

Para una entrevista, prepara una explicación de dos minutos: problema → datos → método → comparación → resultado → límites. Identifica qué código te ayudó a generar IA y demuestra que entiendes cada decisión.


## Mejoras del 11 de octubre de 2026

Entrenamos una red neuronal pequeña con imágenes reales de dígitos. La demo pública utiliza sus pesos entrenados. En una partición obtuvo 98,06%, igual que la regresión logística. En tres semillas el promedio fue 96,48%. Mostrar una red neuronal no significa que siempre sea mejor. Ejecuta neural_train.py y revisa los errores en la demo neural-demo.html.

La página evidence.html conecta resultados, pruebas y código para un revisor. RECRUITER_GUIDE.md relaciona capacidades demostradas con áreas de tus candidaturas y señala lo que estos proyectos todavía no demuestran.
