# Datos / corpus

Material fuente del chatbot: las 7 guías clínicas de VIH de **GeSIDA**.

- **`pdfs/`** — los PDFs originales. **No se versionan en este repositorio** (están en
  `.gitignore`): son obra de GeSIDA, así que se mantienen solo en local y no se redistribuyen.
  Se descargan de GeSIDA si se necesitan.
- **`markdown/`** — los PDFs convertidos a Markdown; es el **corpus real** del que se hace el
  chunking (`../chunks/`).
- **`textos/`** — extracciones de texto plano de los originales (provenance / respaldo).

## Cómo se generaron los Markdown

Cada `.md` se obtuvo del PDF correspondiente con **código de extracción generado con Claude
Code y adaptado a cada PDF por separado**. No hay un único script genérico: cada guía tiene
particularidades de maquetación (tablas, numeración de secciones, notas al pie, columnas) que
no son extrapolables a las demás, de modo que la extracción se ajustó documento a documento.

La conversión se apoya en la librería de **transcripción** PDF→Markdown `pymupdf4llm`, que
**copia el texto literal, no lo genera** (sin modelos de visión que "interpreten" la
estructura). Es una decisión deliberada: el objetivo nº 1 del proyecto es no alucinar, y eso
empieza por que el corpus sea **fiel al original**, sin que un modelo "reescriba" el contenido
de las guías en el paso de ingesta.

El **prompt** que se usó para dirigir esta conversión está en [`prompt.txt`](prompt.txt). Marca
la fidelidad como regla absoluta (no parafrasear/resumir/reconstruir; omitir con una marca
`> _[... omitido — consultar PDF original]_` lo que no se pueda extraer con garantía) y define
el proceso: inspeccionar el PDF para hallar sus particularidades → construir un script
`pymupdf4llm` adaptado a ellas → validar encabezados y muestras → iterar hasta que sea fiel.

> Los derechos y la autoría del contenido pertenecen a GeSIDA; aquí solo se transcribe para uso
> del prototipo.
