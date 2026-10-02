# Resumen del estado del proyecto Call_Me_Maybe

## Que es el proyecto

Call_Me_Maybe consiste en crear una herramienta que convierte peticiones en
lenguaje natural en llamadas a funciones estructuradas en JSON. Por ejemplo,
con la peticion "What is the sum of 2 and 3?" el programa debe producir algo
como:

```json
{"prompt": "What is the sum of 2 and 3?", "fn_name": "fn_add_numbers", "args": {"a": 2.0, "b": 3.0}}
```

El punto central del proyecto es la **decodificacion restringida**: en lugar de
esperar a que el LLM genere un JSON correcto por si solo (fiable solo ~30% del
tiempo con un modelo pequeno), se modifica cada token a generar para que la
salida sea siempre un JSON valido que cumple el esquema requerido.

## Que hay implementado ahora mismo

Estructura del repositorio:

- `src/`: tu implementacion.
  - `__main__.py`: punto de entrada. Lee los JSON de entrada, carga el modelo,
    procesa cada prompt y escribe el resultado.
  - `function_caller.py`: `FunctionCaller`. Selecciona la funcion y pide al
    decodificador que genere los argumentos.
  - `constrained_decoder.py`: maquina de estados que restringe los tokens
    permitidos en cada paso para producir un JSON con forma conocida.
  - `prompt_builder.py`: construye los prompts que se le pasan al modelo.
  - `models.py`: modelos pydantic (`FunctionDefinition`, `FunctionCall`, etc.).
  - `tools.py`: carga y descarga de JSON, validacion basica.
  - `vocab.py`: `VocabIndex`, mapea entre tokens (strings) e IDs.
- `llm_sdk/`: el SDK proporcionado con `Small_LLM_Model`.
- `moulinette/`: la moulinette de correccion (genera ejercicios y corrige).
- `data/input/`: `functions_definition.json` y `function_calling_tests.json`.
- `Makefile`, `pyproject.toml`, `uv.lock`, `.gitignore`.
- `README.md` existe pero esta vacio.

## Que funciona y que no

Funciona en parte:

- La tuberia principal existe: leer entrada, elegir funcion, generar JSON,
  escribir salida.
- La idea de restringir tokens por estado esta bien encaminada.
- Los modelos pydantic estan definidos.

Problemas encontrados (bloqueantes):

1. En `constrained_decoder.py`, `extracted_args` solo se rellena para valores
   numericos. Los argumentos de tipo string nunca se guardan, asi que la salida
   quedaria incompleta.
2. En `function_caller.py` se usa `param.type.value` como si `type` fuera un
   Enum, pero en `models.py` `ParameterDefinition.type` es un `str`. Eso da
   error en tiempo de ejecucion.
3. En `prompt_builder.py`, `build_arg_prompt` usa `fn_def.params[...]`, pero el
   campo real se llama `parameters`. Esa funcion esta rota.
4. Uso inconsistente del SDK: `model.encode()` devuelve un tensor 2D, pero en
   el codigo se trata como una lista (`prompt_ids + generated`,
   `torch.tensor([prompt_ids])`). De ahi viene el error comentado en
   `__main__.py`: "only integer tensors of a single element can be converted
   to an index".
5. `constrained_decoder.py` instancia `Small_LLM_Model()` al importar el
   modulo, lo que carga el modelo cada vez que se importa.
6. En `src/` se importan `torch` y se usa el modelo de forma directa; el
   subject prohibe usar pytorch/transformers/huggingface directamente y insta
   a apoyarse solo en `llm_sdk`.
7. El Makefile no tiene el target obligatorio `debug`, y `MYPY_EXCLUDE` tiene
   `--exclude` duplicado.
8. `README.md` obligatorio esta vacio.
9. No hay tests con pytest/unittest.
10. `tools.json_reader` devuelve `None` ante JSON invalido en vez de un error
    claro; no hay manejo de tipos `boolean`/`integer` en el decodificador.
11. La salida debe contener exactamente las claves `prompt`, `fn_name`,
    `args`, sin texto libre ni claves extra.

## Que debes reaprender/saber para hablar con propiedad de tu codigo

- Python tipado: `typing`, anotaciones de parametros y retorno, `mypy`,
  `flake8`, y por que el subject exige pasar ambos.
- Pydantic: `BaseModel`, `field_validator`, `Field`, y para que sirve validar
  los datos de entrada con ello.
- JSON y esquema: que significa "cumplir el esquema" (claves exactas, tipos,
  argumentos obligatorios) y como validar que la salida cumple.
- Tokenizacion: que un token no es una palabra, que los nombres de funcion se
  parten en varios tokens, y como usar `vocab.json` para mapear token <-> id.
- LLMs y logits: que es un logit, que es el argmax/greedy decoding, y que cada
  paso genera un token a partir de los anteriores.
- Decodificacion restringida: la idea de poner a -infinito los logits de los
  tokens invalidos y quedarse con los validos; como construir el conjunto de
  tokens validos en cada estado.
- El API de `llm_sdk`: `encode`, `decode`, `get_logits_from_input_ids`,
  `get_path_to_vocab_file`, y que NO se pueden usar metodos privados.
- uv y entornos virtuales: `uv sync`, `uv run python -m src`.
- Makefile: reglas `install`, `run`, `debug`, `clean`, `lint`, `lint-strict`.

## Que te falta por aprender/saber para terminar el proyecto

- Arreglar el manejo de tipos: que `type` es `str` y como mapear
  "number"/"string"/"boolean"/"integer" a restricciones de tokens.
- Extraer tambien los argumentos string (y idealmente boolean/integer) en el
  decodificador, y verificar que el JSON generado parsea con `json.loads`.
- Unificar el uso del SDK: trabajar siempre con listas de IDs o tensores de
  forma coherente, sin mezclar.
- Instanciar el modelo una sola vez y pasarlo a quien lo necesite.
- Elegir la funcion con el LLM de forma robusta (restringida a los nombres de
  funcion validos), no con heuristicas.
- Manejo de errores: archivos inexistentes, JSON invalido, prompts ambiguos, y
  mensajes claros sin que el programa pete.
- Cumplir el formato de salida exacto y validarlo antes de escribir.
- Escribir el `README.md` con las secciones obligatorias (descripcion,
  instrucciones, recursos y uso de IA, algoritmo, decisiones de diseno,
  rendimiento, retos, estrategia de pruebas, ejemplos).
- Anadir tests con pytest y el target `debug` al Makefile.
- Probar con la moulinette (`grade_student_answers`) y con casos limite:
  cadenas vacias, numeros grandes, caracteres especiales, prompts ambiguos.

## Orden sugerido de trabajo

1. Entender y depurar el flujo actual (`__main__` -> `FunctionCaller` ->
   `constrained_decoder`) con un solo prompt.
2. Arreglar los bugs de tipos y nombres de campos.
3. Reescribir el decodificador para que extraiga todos los tipos y produzca
   JSON valido al 100%.
4. Limpiar imports y el uso directo del modelo.
5. Makefile (`debug`, lint), tests y README.
6. Validar con la moulinette y medir tiempo/precision.
