# Datos / corpus

Material fuente del chatbot: las 7 guías clínicas de VIH de **GeSIDA**.

- **`pdfs/`** — los PDFs originales, tal cual se publican.
- **`markdown/`** — los PDFs convertidos a Markdown; es el **corpus real** del que se hace el
  chunking (`../chunks/`).
- **`textos/`** — extracciones de texto plano de los originales (provenance / respaldo).

## Cómo se generaron los Markdown

Cada `.md` se obtuvo del PDF correspondiente con **código de extracción generado con Claude
Code y adaptado a cada PDF por separado**. No hay un único script genérico: cada guía tiene
particularidades de maquetación (tablas, numeración de secciones, notas al pie, columnas) que
no son extrapolables a las demás, de modo que la extracción se ajustó documento a documento.

La conversión se apoya en librerías de **transcripción** PDF→Markdown (p. ej. `pymupdf4llm`)
que **copian el texto literal, no lo generan**. Es una decisión deliberada: el objetivo nº 1
del proyecto es no alucinar, y eso empieza por que el corpus sea **fiel al original**, sin que
un modelo "reescriba" el contenido de las guías en el paso de ingesta.

> Los derechos y la autoría del contenido pertenecen a GeSIDA; aquí solo se transcribe para uso
> del prototipo.
