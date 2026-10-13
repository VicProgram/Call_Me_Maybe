# TODO — Arreglar los fallos de `grade_student_answers` (set private)

Estado actual: **4/11 (36.4%)**. Objetivo: 11/11 sin romper la parte pública
ni las garantías del decoder.

> Este documento es solo un plan. No se ha modificado ningún fichero de `src/`.

---

## 0. Contexto rápido (por qué falla cada test)

| Test | Qué pasa | Causa | ¿Bug o modelo? |
|---|---|---|---|
| 1, 2 | `{"a": 3}` en vez de `3.0`; `assert isinstance(a, float)` revienta | `number` se serializa como `int` | **Bug tuyo** |
| 7 | falta el `)` final en la query | el token terminador `)"` se descarta entero | **Bug del decoder** |
| 8 | `" /home/user/data.json"` con espacio inicial | el modelo mete un espacio | Modelo (fácil de normalizar) |
| 9 | `C:/Users/...` en vez de `C:\Users\...` | el mask prohíbe `\` en strings | Diseño del decoder |
| 10 | llama `fn_read_file` en vez de `fn_format_template` | selección de función errónea | Modelo + prompt |
| 11 | igual que 10, y el valor tiene `{}` y `"` | selección + caracteres imposibles | Modelo + diseño |

Criterio del grader (`moulinette/moulinette/__main__.py`):
- empareja **por posición** (`zip`) y exige el **prompt exacto**;
- `fn(**fn_params)` con asserts de tipo;
- comparación de salida con `!=` estricto.

---

## 1. Quick wins (bajo riesgo, alto impacto)

- [ ] **T1. Coaccionar `number` → `float`** — *Impacto: tests 1, 2 (+2)*
  - Dónde: `src/function_caller.py` → `FunctionCaller.resolve` (tras validar tipos).
  - Qué: al construir `parameters`, si el tipo declarado es `"number"`, hacer
    `float(value)`. Así se serializa `3.0` y el `assert isinstance(a, float)` pasa.
  - Cuidado: NO coaccionar `"integer"` (test 5 usa `years: int`). No tocar `bool`.
  - Verificar: reejecutar y ver `{"a": 3.0, "b": 5.0}` en el output.
  - Esfuerzo: ~15 min · Riesgo: muy bajo.

- [ ] **T2. Arreglar el terminador de string** — *Impacto: test 7 (+1)*
  - Dónde: `src/constrained_decoder.py` → `generate_string` (líneas ~181-182).
  - Qué: al elegir un token terminador, **no tirarlo entero**. Decodificarlo,
    quedarte con el texto anterior a la primera `"` / `\n` y añadirlo al
    contenido (re-encodeando ese prefijo como tokens).
  - Ejemplo real: en el vocab de Qwen existe el token `)"` (id `9940`), por eso
    se pierde el `)`.
  - Alternativa: terminar solo si el token es exactamente `"`/`\n`; si trae
    prefijo, emitir el prefijo y cerrar después.
  - Verificar: la query debe quedar `INSERT INTO logs VALUES (1, 2, 3)`.
  - Esfuerzo: ~1 h · Riesgo: medio (tocar generación de strings; añadir test).

---

## 2. Valores que no se pueden generar (tests 8, 9, 10, 11)

Estos NO son solo "el modelo se equivoca": hay caracteres que el decoder
**prohíbe por diseño**.

- [ ] **T3. Permitir o evitar `{ } [ ] \` en strings** — *Impacto: tests 9, 10, 11*
  - Dónde: `src/constrained_decoder.py:125` (filtro `if not any(char in token for char in "{}[]\\")`).
  - Opción A (recomendada): **extraer el valor de forma determinista** desde el
    `prompt` original, en lugar de generarlo token a token.
  - Opción B (más trabajo y riesgo): permitir `\` y llaves en el mask y manejar
    el escapado JSON correctamente (`\\` en el texto → `\` al hacer `json.loads`).
  - Nota: aunque arregles la selección (T5), 10 y 11 seguirían fallando sin esto.

- [ ] **T4. Extracción determinista de argumentos desde el prompt** — *Impacto: tests 7, 8, 9, 10, 11*
  - Qué: heurísticas/parsers que copian el valor literal del enunciado:
    - texto entre comillas simples/dobles (tests 7, 10 implícito),
    - texto tras un prefijo tipo `Format template:` (tests 10, 11),
    - rutas con `/` o `\` (tests 8, 9),
    - números (tests 1-6).
  - Dónde: un módulo nuevo (`src/value_extractor.py`) o dentro de `function_caller.py`.
  - Uso: LLM solo para **elegir la función**; los valores se extraen del prompt.
  - Verificar: `.strip()` en todos los strings (arregla el espacio del test 8).
  - Esfuerzo: ~2-4 h · Riesgo: medio (cuidado con sobreajustar al test set).

---

## 3. Selección de función

- [ ] **T5. Mejorar `select_function_name` / prompt de selección** — *Impacto: tests 10, 11*
  - Dónde: `src/prompt_builder.py` (`fun_selection`) y
    `src/constrained_decoder.py` (`select_function_name`, ~285-318).
  - Ideas:
    - prompt más explícito: "responde solo con el nombre exacto de la lista";
    - incluir descripciones con más peso / formato consistente;
    - few-shot con 1-2 ejemplos;
    - comparar mejor el scoring (primer token significativo, contexto propio).
  - Alternativa barata: capa de routing por palabras clave
    (`format template` → `fn_format_template`).
  - Verificar: "Format template: ..." debe elegir `fn_format_template`.
  - Esfuerzo: ~1-3 h · Riesgo: medio.

---

## 4. Calidad y regresión

- [ ] **T6. Tests unitarios** para cada fix (terminador, coacción `number`,
      extractor). Añadir a `tests/`.
- [ ] **T7. `make lint` y `make test`** en verde.
- [ ] **T8. Actualizar `README.md`**: la tabla de la línea 142 (prohibición de
      `{ } [ ] \`) y la sección "Challenges encountered" deben reflejar el nuevo
      comportamiento.
- [ ] **T9. Verificación final end-to-end**:
  - `uv run python -m src --input <dir_private>` (o `make run` con el set privado),
  - `uv run python -m moulinette grade_student_answers --set private --student_answer_path ../data/output/function_calling_results.json`,
  - confirmar 11/11 y que el set público sigue OK.

---

## 5. Orden recomendado / hitos

| Hito | Tareas | Score esperado | Esfuerzo |
|---|---|---|---|
| M1 | T1 | 6/11 | ~15 min |
| M2 | T1 + T2 | 7/11 | ~1-2 h |
| M3 | + T4 (strip + extracción) | 9/11 | ~4-6 h |
| M4 | + T3/T5 (plantillas y selección) | 11/11 | ~7-12 h |
| M5 | + T6-T9 (calidad, docs) | 11/11 limpio | +2-4 h |

---

## 6. Cómo decidir si hacerlo (para valorar en casa)

- **Hazlo seguro:** T1 (coacción a `float`) → 2 tests gratis, cero riesgo,
  corrige un bug de tipos real. T2 es un bug de verdad del decoder; merece la pena.
- **Valora:** T3/T4. Funcionan, pero la extracción determinista puede leerse como
  "hardcodear el test set". Si el evaluador premia generalidad, justifica el
  extractor como *parser-assisted extraction* (genérico por patrón de tipo) y no
  como casos concretos.
- **Pregunta clave:** ¿puntúa más el score final o la limpieza del diseño?
  - Si pesa el score → M1→M4.
  - Si pesa la elegancia → M1+M2 y, como mucho, T4 generalizado.

### Riesgos globales
- Sobreajustar al set privado (que no ves) usando heurísticas demasiado concretas.
- Tocar el mask de strings y romper el escapado JSON.
- Regresión en el set público / en la garantía de "JSON sempre válido".

---

## 7. Referencias de código

- Tipos de valor / validación: `src/function_caller.py:8-28`
- Mask de strings: `src/constrained_decoder.py:123-128`
- Generación de string: `src/constrained_decoder.py:157-188`
- Selección de función: `src/constrained_decoder.py:285-318`
- Prompt de selección: `src/prompt_builder.py:4-36`
- Grader: `moulinette/moulinette/__main__.py:128-174`
- Funciones y asserts: `moulinette/moulinette/functions_definition.py`
