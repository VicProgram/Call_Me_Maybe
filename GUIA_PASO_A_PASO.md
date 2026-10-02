# Call Me Maybe - Guia de Aprendizaje Completa

## Como usar esta guia

Esta guia esta escrita como un libro de texto. Cada capitulo explica primero
el concepto, luego el por que existe, despues el como se hace, y finalmente
muestra un ejemplo de codigo. Puedes leerla de principio a fin o saltar al
capitulo que necesites.

---

## Capitulo 1: Entendiendo el Problema

### 1.1 Que es un LLM y como piensa

Un LLM (Large Language Model) es un programa que ha leido mucho texto y ha
aprendido a predecir que palabra sigue despues de otra. Cuando le escribes
"Hola, como", el modelo piensa que lo mas probable es que siga "estas?".

Pero el modelo no trabaja con palabras. Trabaja con **tokens**, que son
numeros enteros. Cada numero representa un trozo de texto. El modelo recibe
una lista de numeros y devuelve una lista de puntuaciones (llamadas **logits**)
que indican que tan probable es que cada token del vocabulario sea el siguiente.

**Por que importa**: para controlar lo que genera el modelo, necesitas
entender que decide en cada paso y como influir en esa decision.

### 1.2 El problema de generar JSON

JSON es un formato estricto. Requiere comillas, comas, llaves y tipos de datos
especificos. Un modelo pequeno (como Qwen3-0.6B, con solo 600 millones de
parametros) no es muy bueno siguiendo estas reglas. Si le pides que genere
JSON, a menudo:

- Se olvida de cerrar una comilla
- Anade texto extra antes o despues del JSON
- Inventa claves que no existen
- Mezcla tipos de datos (pone un string donde deberia ir un numero)

**El dato clave del subject**: un modelo pequeno genera JSON valido solo el
30% de las veces. Pero con la tecnica correcta, podemos llegar al 100%.

### 1.3 La solucion: Decodificacion Restringida

La idea es simple pero poderosa: en lugar de dejar al modelo elegir
libremente, le damos un menu de opciones validas en cada paso.

**Analogia**: imagina que estas llenando un formulario. En cada campo, solo
puedes escribir lo que el campo acepta. Si es un campo de fecha, solo puedes
escribir numeros y barras. Si es un campo de texto, puedes escribir cualquier
letra. El formulario te guia para que no te equivocas.

En decodificacion restringida, el "formulario" es una maquina de estados que
sabe que parte del JSON se esta generando en cada momento y que caracteres
son validos en ese punto.

**Por que funciona**: el modelo sigue decidiendo que generar (que funcion
llamar, que valores poner), pero solo entre opciones que mantienen el JSON
valido. Es como un copiloto que te dice "por aqui no" cuando te equivocas de
camino.

---

## Capitulo 2: Tokenizacion - El Puente entre Texto y Numeros

### 2.1 Por que los modelos no usan palabras

Las palabras son ambiguas y numerosas. Un modelo tendria que conocer millones
de palabras en multiples idiomas. En su lugar, los modelos usan **tokens**,
que son trozos mas pequenos de texto.

Un token puede ser:

- Una palabra completa: "casa"
- Una parte de palabra: "casa" + "s" = "casas"
- Un solo caracter: "a"
- Un espacio: " " (los espacios son importantes en JSON)

### 2.2 BPE (Byte-Pair Encoding)

El algoritmo mas comun para crear tokens es BPE. Funciona asi:

1. Empieza con caracteres individuales.
2. Busca las parejas de caracteres que mas juntas aparecen en el texto.
3. Fusiona esas parejas en un nuevo token.
4. Repite hasta tener el numero deseado de tokens.

**Ejemplo practico**:

```
Texto de entrenamiento: "casa casas casar"
Paso 1: ["c", "a", "s", "a", " ", "c", "a", "s", "a", "s", ...]
Paso 2: "ca" aparece mucho -> crear token "ca"
Paso 3: "cas" aparece mucho -> crear token "cas"
Paso 4: "casa" aparece mucho -> crear token "casa"
Resultado: "casa" es un solo token, "casas" es "casa" + "s"
```

### 2.3 El truco de los bytes a unicode

Aquí viene un detalle tecnico importante. El archivo `vocab.json` del modelo
no guarda los tokens como texto normal. Usa un mapeo de bytes a caracteres
unicode. Esto se hace porque no todos los bytes son caracteres validos en JSON.

**Por que**: un byte puede tener cualquier valor de 0 a 255. Pero muchos de
 esos valores no son caracteres imprimibles. Para guardarlos en un archivo
 JSON, se mapean a caracteres unicode que si son imprimibles.

**El mapeo**:

```
Byte 32 (espacio) -> Caracter "Ġ"
Byte 10 (salto de linea) -> Caracter "Ċ"
Byte 65 (A) -> Caracter "A" (se queda igual)
```

**Como revertirlo**: para saber que texto real produce un token, necesitas
reconstruir el mapeo inverso y decodificar los bytes.

### 2.4 Por que necesitamos saber esto

En decodificacion restringida, el modelo trabaja con IDs de tokens. Pero
nosotros queremos razonar sobre caracteres (por ejemplo, "el siguiente
caracter debe ser una comilla"). Necesitamos una forma de convertir entre IDs
y texto real.

**Solucion**: una clase `Vocabulary` que cargue el `vocab.json` y nos de el
texto real de cualquier token.

---

## Capitulo 3: Logits - La Voz del Modelo

### 3.1 Que son los logits

Cuando el modelo procesa una secuencia de tokens, produce un array de
numeros. Cada numero es un **logit** (puntuacion) para un token del
vocabulario. El token con mayor logit es el que el modelo "quiere" generar.

**Ejemplo**:

```
Prompt: "The cat"
Logits: [0.1, 0.05, 0.7, 0.1, 0.05]
Tokens: ["sat", "ran", "slept", "ate", "jumped"]
```

En este ejemplo, el modelo prefiere "slept" (logit 0.7) sobre las otras
opciones.

### 3.2 Como usar los logits

Normalmente, se elige el token con mayor logit (esto se llama **greedy
decoding**). Pero en decodificacion restringida, hacemos algo diferente:

1. Recibimos los logits del modelo.
2. Identificamos que tokens son validos segun nuestra maquina de estados.
3. Ponemos a `-infinito` los logits de los tokens invalidos.
4. Elegimos el token con mayor logit entre los validos.

**Por que `-infinito`**: asi el token invalido nunca sera elegido, porque
siempre habra un valido con mayor puntuacion.

### 3.3 Softmax (opcional)

Los logits se pueden convertir en probabilidades usando la funcion softmax:

```
probabilidad[i] = exp(logit[i]) / suma(exp(logit[j]) para todo j)
```

En decodificacion restringida no necesitas softmax, porque solo te importa
que token tiene el mayor logit. Pero es bueno saber que existe.

---

## Capitulo 4: La Maquina de Estados - El Corazon del Proyecto

### 4.1 Que es una maquina de estados

Una maquina de estados es un modelo que tiene:

- **Estados**: situaciones en las que puede estar.
- **Transiciones**: reglas para pasar de un estado a otro.
- **Entradas**: lo que hace que cambie de estado.

**Analogia**: un semaforo. Tiene tres estados (rojo, amarillo, verde). Solo
puede pasar de rojo a verde, de verde a amarillo, y de amarillo a rojo. No
puede pasar de rojo a amarillo directamente.

### 4.2 Por que una maquina de estados para JSON

El JSON que generamos tiene una estructura fija:

```json
{"name": "<FUNCION>", "parameters": {"<CLAVE>": <VALOR>, ...}}
```

Solo hay dos partes libres: el nombre de la funcion y los valores de los
parametros. Todo lo demas son caracteres fijos (llaves, comillas, comas,
dos puntos).

Una maquina de estados puede recorrer esta estructura y, en cada momento,
saber que caracteres son validos.

### 4.3 Los estados de nuestra maquina

| Estado | Que hace | Caracteres validos |
|--------|----------|-------------------|
| PREFIX | Emite el inicio del JSON | `{`, `"`, `n`, `a`, `m`, `e`, `"`, `:`, ` `, `"` |
| NAME | Emite el nombre de la funcion | Letras que continuan un nombre valido |
| AFTER_NAME | Emite el puente hacia los parametros | `"`, `,`, ` `, `"`, `p`, `a`, `r`, `a`, `m`, `e`, `t`, `e`, `r`, `s`, `"`, `:`, ` `, `{` |
| PARAM_KEY | Emite la clave de un parametro | `"`, `a`, `"`, `:`, ` ` |
| VALUE_NUMBER | Emite un numero | Digitos, `-`, `.`, `,`, `}` |
| VALUE_STRING | Emite una string | Caracteres imprimibles, `"` |
| VALUE_BOOLEAN | Emite true o false | `t`, `r`, `u`, `e`, `f`, `a`, `l`, `s`, `e`, `,`, `}` |
| SEPARATOR | Emite la coma entre parametros | `,`, ` ` |
| SUFFIX | Emite el cierre del JSON | `}`, `}` |
| DONE | Ya termino | (nada) |

### 4.4 Como funciona en la practica

Supongamos que queremos generar:

```json
{"name": "fn_add", "parameters": {"a": 1, "b": 2}}
```

La maquina avanza asi:

1. **PREFIX**: emite `{"name": "` caracter por caracter.
2. **NAME**: el modelo elige `fn_add` (entre las funciones disponibles).
3. **AFTER_NAME**: emite `", "parameters": {`.
4. **PARAM_KEY**: emite `"a": `.
5. **VALUE_NUMBER**: el modelo elige `1`.
6. **SEPARATOR**: emite `, `.
7. **PARAM_KEY**: emite `"b": `.
8. **VALUE_NUMBER**: el modelo elige `2`.
9. **SUFFIX**: emite `}}`.
10. **DONE**: terminado.

En cada paso, la maquina dice "solo estos caracteres son validos". El modelo
elige entre ellos.

### 4.5 El metodo accepts()

Este metodo es crucial. Permite preguntar: "si te doy este token completo,
lo aceptarias?".

**Por que es necesario**: los tokens pueden tener multiples caracteres. Un
token podria ser `"a": ` (4 caracteres). Necesitamos saber si esos 4
caracteres son validos en secuencia.

**Como funciona**:

1. Guarda el estado actual.
2. Intenta avanzar caracter por caracter.
3. Si todos son validos, devuelve True.
4. Si alguno no es valido, restaura el estado y devuelve False.
5. Si todos son validos, restaura el estado (no lo compromete).

**Ejemplo**:

```python
constraint = FunctionCallConstraint(functions)
constraint.advance('{')
constraint.advance('"')
constraint.advance('n')
# Ahora estamos en NAME

# Preguntamos si "fn_add" seria valido
if constraint.accepts("fn_add"):
    # Si, es valido
    pass
```

---

## Capitulo 5: El Decoder - Uniendo Todo

### 5.1 Que hace el decoder

El decoder es el bucle que:

1. Pide logits al modelo.
2. Pregunta a la maquina que caracteres son validos.
3. Filtra los tokens: solo los que la maquina acepta.
4. Elige el token con mayor logit entre los validos.
5. Consume ese token en la maquina.
6. Repite hasta que la maquina este completa.

### 5.2 El algoritmo paso a paso

```
1. constraint = FunctionCallConstraint(functions)
2. input_ids = encode(prompt)
3. mientras no constraint.is_complete():
4.     logits = model.get_logits_from_input_ids(input_ids)
5.     legal_tokens = []
6.     para cada token_id en vocabulario:
7.         text = vocab.token_text(token_id)
8.         si constraint.accepts(text):
9.             legal_tokens.append(token_id)
10.    best_token = argmax(logits[legal_tokens])
11.    text = vocab.token_text(best_token)
12.    para cada ch en text:
13.        constraint.advance(ch)
14.    input_ids.append(best_token)
15. devolver generated_text
```

### 5.3 Por que inyectar el modelo

En lugar de hacer `from llm_sdk import Small_LLM_Model` dentro del decoder,
recibimos una funcion `logits_fn` como argumento.

**Ventajas**:

- **Testeable**: puedes pasar un modelo falso que devuelva logits
  predecibles.
- **Flexible**: puedes cambiar el modelo sin tocar el decoder.
- **Rapido**: los tests no necesitan cargar el modelo real (que pesa 1.5 GB).

**Ejemplo**:

```python
# En produccion
decoder = ConstrainedDecoder(
    functions=functions,
    vocabulary=vocab,
    logits_fn=lambda ids: model.get_logits_from_input_ids(ids),
    encode_fn=lambda text: model.encode(text).flatten().tolist(),
)

# En tests
decoder = ConstrainedDecoder(
    functions=functions,
    vocabulary=fake_vocab,
    logits_fn=lambda ids: [1.0] * len(fake_vocab),  # siempre el mismo
    encode_fn=lambda text: [0],
)
```

---

## Capitulo 6: Pydantic - Validacion de Datos

### 6.1 Que es pydantic

Pydantic es una libreria que valida datos en Python. Defines un modelo con
tipos, y pydantic se encarga de que los datos cumplan esos tipos.

**Por que el subject lo exige**: "Todas las clases deben usar pydantic para
validacion." Esto garantiza que los datos de entrada y salida son correctos.

### 6.2 Como funciona

```python
from pydantic import BaseModel

class Persona(BaseModel):
    nombre: str
    edad: int

# Funciona
p = Persona(nombre="Ana", edad=25)

# Falla: edad debe ser int
p = Persona(nombre="Ana", edad="25")  # Error de validacion
```

### 6.3 En nuestro proyecto

Usamos pydantic para:

- **FunctionDefinition**: valida que las funciones de entrada tengan los
  campos correctos (name, description, parameters, returns).
- **FunctionCall**: valida que la salida tenga los campos correctos
  (prompt, fn_name, args).

**Ejemplo de validacion**:

```python
# Esto funciona
fn = FunctionDefinition(
    name="fn_add",
    description="Add two numbers",
    parameters={"a": ParameterDefinition(type="number")},
    returns={"type": "number"},
)

# Esto falla: falta description
fn = FunctionDefinition(
    name="fn_add",
    parameters={"a": ParameterDefinition(type="number")},
    returns={"type": "number"},
)  # Error: campo requerido faltante
```

---

## Capitulo 7: Manejo de Errores

### 7.1 Por que es importante

El subject dice: "Todos los errores deben gestionarse correctamente. El
programa nunca debe fallar de forma inesperada y siempre debe proporcionar
mensajes de error claros."

Esto significa que si el usuario:

- Borra el archivo de entrada
- Pone un JSON malformado
- Pide algo imposible

El programa debe decir "Error: el archivo X no existe" en vez de petar con un
traceback.

### 7.2 Tipos de errores que manejamos

| Error | Causa | Mensaje al usuario |
|-------|-------|-------------------|
| FileNotFoundError | El archivo de entrada no existe | "Error: Functions file not found: data/input/functions_definition.json" |
| ValueError (JSON) | El JSON esta malformado | "Error: Invalid JSON in functions_definition.json" |
| ValueError (schema) | El JSON no tiene la estructura correcta | "Error: Invalid function definition at index 2" |
| RuntimeError (decoder) | El decoder no puede generar JSON valido | "Error: No legal token at step 42" |

### 7.3 Como implementarlo

```python
def main() -> int:
    args = parse_args()

    try:
        functions = load_functions(args.input / "functions_definition.json")
        prompts = load_prompts(args.input / "function_calling_tests.json")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # ... resto del programa
    return 0
```

---

## Capitulo 8: Tests - Verificando que Todo Funciona

### 8.1 Por que testear

Los tests son como un examen que le haces a tu codigo. Si el test pasa, el
codigo hace lo que esperas. Si falla, sabes que algo esta mal.

**En este proyecto**: los tests son especialmente importantes porque el
modelo es lento (tarda en cargar) y no siempre se puede probar con el modelo
real.

### 8.2 Tipos de tests

1. **Tests unitarios**: prueban una pieza pequena (ej: la maquina de estados).
2. **Tests de integracion**: prueban como trabajan varias piezas juntas.
3. **Tests end-to-end**: prueban el programa completo con el modelo real.

### 8.3 Tests de la maquina de estados (sin modelo)

Estos tests son rapidos y no necesitan el modelo. Verifican que la maquina
acepta JSON validos y rechaza JSON invalidos.

**Ejemplo**:

```python
def test_accepts_valid_json():
    c = FunctionCallConstraint(functions)
    assert c.accepts('{"name": "fn_add", "parameters": {"a": 1, "b": 2}}')

def test_rejects_invalid_json():
    c = FunctionCallConstraint(functions)
    assert not c.accepts('{"name": "fn_add", "parameters": {"a": 1, "b": 2}}x')
```

### 8.4 Tests del decoder (con modelo falso)

Estos tests verifican que el decoder produce JSON valido, usando un modelo
falso que devuelve logits predecibles.

**Ejemplo**:

```python
def test_decoder_produces_valid_json():
    decoder = ConstrainedDecoder(functions, fake_vocab, fake_logits, fake_encode)
    result = decoder.decode("test")
    assert result.startswith('{"')
    assert result.endswith('}}')
```

---

## Capitulo 9: El README - Contando tu Historia

### 9.1 Por que es importante

El README es lo primero que ve alguien que abre tu proyecto. Debe explicar:

- Que hace el proyecto
- Como instalarlo
- Como ejecutarlo
- Como funciona por dentro
- Que decisiones tomaste
- Que problemas encontraste

### 9.2 Secciones obligatorias (segun el subject)

1. **Primera linea en cursiva**: `*Este proyecto ha sido creado como parte
   del curriculo de 42 por <login>.*`
2. **Descripcion**: que hace el proyecto.
3. **Instrucciones**: como instalar y ejecutar.
4. **Recursos**: enlaces a documentacion + seccion "Uso de IA".
5. **Explicacion del algoritmo**: decodificacion restringida.
6. **Decisiones de diseno**: por que separaste asi el codigo.
7. **Analisis de rendimiento**: precision, velocidad, fiabilidad.
8. **Retos encontrados**: que problemas tuviste y como los resolviste.
9. **Estrategia de pruebas**: como validaste el proyecto.
10. **Ejemplos de uso**: comandos de ejemplo.

### 9.3 La seccion "Uso de IA"

El subject pide que expliques para que tareas usaste IA. Esto es importante
porque quieren saber que entiendes lo que hiciste, no que copiaste y pegaste.

**Ejemplo de como escribirla**:

```
### Uso de IA

Se uso IA como asistente para:
- Explicar conceptos de tokenizacion y decodificacion
- Revisar el codigo
- Mejorar la documentacion

Todas las decisiones de diseno y la implementacion fueron realizadas por mi.
Las sugerencias de IA fueron revisadas, adaptadas y validadas antes de usarlas.
```

---

## Capitulo 10: El Makefile - Automatizacion

### 10.1 Que es un Makefile

Un Makefile es un archivo que define comandos (targets) que puedes ejecutar
con `make <target>`. Es como tener accesos rapidos para las tareas comunes.

### 10.2 Targets obligatorios

| Target | Que hace | Comando tipico |
|--------|----------|----------------|
| `install` | Instala dependencias | `uv sync` |
| `run` | Ejecuta el programa | `uv run python -m src` |
| `debug` | Ejecuta con debugger | `uv run python -m pdb -m src` |
| `lint` | Verifica estilo y tipos | `flake8 . && mpy .` |
| `lint-strict` | Verificacion estricta | `flake8 . && mpy . --strict` |
| `clean` | Limpia archivos temporales | `rm -rf __pycache__ .mypy_cache` |

### 10.3 Por que `uv run` en vez de `python`

`uv run python -m src` hace dos cosas:

1. Asegura que el entorno virtual este activo.
2. Ejecuta el comando dentro de ese entorno.

Si usas solo `python`, podrias estar usando el Python del sistema, no el del
proyecto.

---

## Capitulo 11: Validacion con la Moulinette

### 11.1 Que es la moulinette

La moulinette es el programa que corrige tu proyecto. Genera ejercicios
(prompts y funciones) y compara tu salida con la esperada.

### 11.2 Como usarla

```bash
# Generar ejercicios (publicos o privados)
cd moulinette
uv run python -m moulinette prepare_exercises --set public

# Corregir tu salida
uv run python -m moulinette grade_student_answers ../data/output/function_calling_results.json
```

### 11.3 Que verifica

- Que el prompt coincide con el esperado.
- Que el nombre de la funcion es correcto.
- que los argumentos son correctos (nombre y tipo).
- Que el resultado de llamar la funcion es correcto.

---

## Capitulo 12: Errores Comunes y Como Evitarlos

### 12.1 Error: `param.type.value`

**Causa**: `ParameterDefinition.type` es un `str`, no un Enum. Los strings no
tienen `.value`.

**Solucion**: usa `param.type` directamente.

### 12.2 Error: `fn_def.params`

**Causa**: el campo se llama `parameters`, no `params`.

**Solucion**: usa `fn_def.parameters`.

### 12.3 Error: `prompt_ids + generated`

**Causa**: `model.encode()` devuelve un Tensor de PyTorch, no una lista. No
puedes sumar un Tensor con una lista.

**Solucion**: convierte a lista primero:
```python
input_ids = model.encode(text).flatten().tolist()
```

### 12.4 Error: JSON invalido

**Causa**: el decoder no restringe bien los caracteres en algun estado.

**Solucion**: revisa que `allowed_next()` devuelva exactamente los caracteres
validos en cada fase, y que `accepts()` verifique el token completo.

### 12.5 Error: el modelo se carga al importar

**Causa**: instanciar `Small_LLM_Model()` a nivel de modulo (fuera de una
funcion) hace que se cargue cada vez que se importa el modulo.

**Solucion**: instanciar el modelo una sola vez en `main()` y pasarlo a quien
lo necesite.

### 12.6 Error: usar torch directamente en src/

**Causa**: el subject prohibe usar pytorch/transformers/huggingface
directamente.

**Solucion**: usa solo `llm_sdk`. Si necesitas manipular tensores, hazlo
dentro de `llm_sdk`, no en tu codigo.

---

## Capitulo 13: Conceptos Avanzados (Opcional)

### 13.1 Por que no usar heuristicas para elegir la funcion

El subject dice: "La funcion a llamar debe elegirse usando el LLM, no con
heuristicas ni ningun otro tipo de magia medieval."

Esto significa que no puedes:

- Buscar palabras clave en el prompt (ej: "suma" -> fn_add).
- Usar expresiones regulares para detectar la intencion.
- Hacer matching de strings.

El modelo debe decidir que funcion llamar basandose en el prompt y las
descripciones de las funciones. La decodificacion restringida garantiza que
el nombre elegido sea uno de los validos, pero la decision es del modelo.

### 13.2 Por que pydantic en toda la frontera

Pydantic valida los datos al crear la instancia. Si el JSON de entrada tiene
un tipo incorrecto o falta un campo, pydantic lanza un error claro. Esto
cumple el requisito de "Todas las clases deben usar pydantic para validacion."

### 13.3 Por que inyectar el modelo

Si el decoder recibe el modelo como argumento (en vez de importarlo), puedes:

- Testear el decoder con un modelo falso (rapido, sin GPU).
- Cambiar el modelo sin modificar el decoder.
- Mantener la dependencia de `llm_sdk` confinada a `pipeline.py`.

### 13.4 El truco de los literales fijos

El JSON de salida tiene una estructura fija:

```
{"name": "<FUNCION>", "parameters": {"<CLAVE>": <VALOR>, ...}}
```

Todo excepto `<FUNCION>` y `<VALOR>` son literales fijos. Esto significa que
en la mayoria de los pasos, solo hay un caracter valido. La maquina de estados
sabe exactamente que literal emitir en cada momento.

### 13.5 El problema de los numeros sin delimitador

Un numero JSON no tiene un delimitador de cierre. No sabes si el numero ha
terminado hasta que ves una coma o una llave. La solucion es:

- Mientras el numero no este completo, solo aceptar digitos, `-` y `.`.
- Cuando el numero este completo, ofrecer `,` o `}` como caracteres validos.
- Si el modelo elige `,` o `}`, el numero ha terminado.

---

## Capitulo 14: Resumen del Orden de Trabajo

1. **Entender el subject**: lee el PDF y anota los requisitos.
2. **Preparar entorno**: `uv sync`.
3. **Estudiar conceptos**: tokenizacion, logits, decodificacion restringida.
4. **Implementar `models.py`**: validacion con pydantic.
5. **Implementar `vocabulary.py`**: mapeo token <-> texto.
6. **Implementar `constraints.py`**: la maquina de estados.
7. **Implementar `decoder.py`**: el bucle de decodificacion.
8. **Implementar `pipeline.py`**: union de todo.
9. **Implementar `io_utils.py` y `__main__.py`**: entrada/salida y CLI.
10. **Makefile**: todos los targets obligatorios.
11. **Tests**: pytest para constraints y decoder.
12. **README.md**: todas las secciones obligatorias.
13. **Validar**: lint, tests, ejecucion real, moulinette.

---

## Capitulo 15: Restricciones del Subject - Lo que Puedes y No Puedes Hacer

### 15.1 Restricciones de lenguaje y dependencias

| Regla | Detalle |
|-------|---------|
| Python 3.10+ | No usar caracteristicas de versiones anteriores |
| flake8 | Todo el codigo debe pasar `flake8 .` |
| mypy | Todo el codigo debe pasar `mypy .` con los flags del subject |
| type hints | Todas las funciones deben tener anotaciones de tipo |
| docstrings | Todas las funciones y clases deben tener docstrings (PEP 257) |
| pydantic | Todas las clases de datos deben usar pydantic |
| numpy | Permitido |
| json | Permitido |
| torch | **PROHIBIDO** en `src/` |
| transformers | **PROHIBIDO** en `src/` |
| huggingface | **PROHIBIDO** en `src/` |
| dspy | **PROHIBIDO** |
| outlines | **PROHIBIDO** |

**Por que**: el subject quiere que aprendas a interactuar con el modelo a
traves del SDK, no que uses las librerias directamente. El SDK es una
abstraccion que simplifica la interaccion.

### 15.2 Restricciones de uso del SDK

| Regla | Detalle |
|-------|---------|
| `get_logits_from_input_ids` | Usar este metodo para obtener logits |
| `get_path_to_vocab_file` | Usar este metodo para obtener el vocabulario |
| `encode` | Usar este metodo para tokenizar texto |
| `decode` | Opcional, para decodificar tokens |
| Metodos privados | **PROHIBIDOS** (nada que empiece con `_`) |

**Por que**: el subject prohibe explicitamente usar metodos privados del SDK.
Esto es para que no dependas de detalles internos que podrian cambiar.

### 15.3 Restricciones de eleccion de funcion

| Regla | Detalle |
|-------|---------|
| Elegir con el LLM | La funcion debe ser elegida por el modelo |
| No heuristicas | No usar matching de palabras clave, regex, etc. |
| No magia medieval | No usar logica hardcodeada |

**Por que**: el subject quiere que el modelo decida que funcion llamar. Tu
trabajo es darle las opciones y dejar que elija, no elegir por el.

### 15.4 Restricciones de formato de salida

| Regla | Detalle |
|-------|---------|
| Claves exactas | `prompt`, `fn_name`, `args` |
| Sin claves extra | No anadir nada mas |
| Sin texto libre | Solo JSON valido |
| Tipos correctos | `number`, `string`, `boolean`, `integer` |
| Todos los argumentos | No olvidar ningun parametro requerido |

### 15.5 Restricciones de manejo de errores

| Regla | Detalle |
|-------|---------|
| Nunca petar | El programa no debe fallar con excepciones no manejadas |
| Mensajes claros | Errores descriptivos para el usuario |
| try-except | Usar bloques try-except para errores esperados |
| Context managers | Usar `with` para archivos |

### 15.6 Restricciones de Makefile

| Target | Obligatorio | Comando |
|--------|-------------|---------|
| `install` | Si | `uv sync` |
| `run` | Si | `uv run python -m src` |
| `debug` | Si | `uv run python -m pdb -m src` |
| `clean` | Si | Limpiar caches |
| `lint` | Si | `flake8 . && mpy .` con flags |
| `lint-strict` | No | `flake8 . && mpy . --strict` |

### 15.7 Restricciones de README

| Seccion | Obligatoria |
|---------|-------------|
| Primera linea en cursiva | Si |
| Descripcion | Si |
| Instrucciones | Si |
| Recursos | Si |
| Uso de IA | Si |
| Explicacion del algoritmo | Si |
| Decisiones de diseno | Si |
| Analisis de rendimiento | Si |
| Retos encontrados | Si |
| Estrategia de pruebas | Si |
| Ejemplos de uso | Si |

### 15.8 Restricciones de entrega

| Regla | Detalle |
|-------|---------|
| `src/` | Debe estar en el repo |
| `pyproject.toml` y `uv.lock` | Deben estar en el repo |
| `llm_sdk/` | Debe estar en el repo (copiado del proporcionado) |
| `data/input/` | Debe estar en el repo (archivos de prueba) |
| `README.md` | Debe estar en el repo |
| `output/` | **NO** incluir en el repo (se genera en evaluacion) |

---

## Capitulo 16: La Interfaz de Linea de Comandos (CLI)

### 16.1 Como se ejecuta el programa

El subject define exactamente como se debe ejecutar:

```bash
uv run python -m src [--input <input_file>] [--output <output_file>]
```

**Que significa cada parte**:

- `uv run`: ejecuta en el entorno virtual del proyecto
- `python -m src`: ejecuta el modulo `src` como script principal
- `--input`: ruta al archivo o directorio de entrada
- `--output`: ruta al archivo de salida

### 16.2 Comportamiento por defecto

Si no se especifican argumentos:

- **Entrada**: `data/input/` (directorio)
- **Salida**: `data/output/function_calling_results.json`

### 16.3 Comportamiento con argumentos

```bash
# Especificar un archivo de entrada personalizado
uv run python -m src --input data/input/mis_tests.json

# Especificar un archivo de salida personalizado
uv run python -m src --output data/output/mis_resultados.json

# Ambos
uv run python -m src --input data/input/mis_tests.json --output data/output/mis_resultados.json
```

### 16.4 Que pasa si --input es un archivo o directorio

- Si es un **directorio**: busca `functions_definition.json` y
  `function_calling_tests.json` dentro de el.
- Si es un **archivo**: lo usa como archivo de tests, y busca
  `functions_definition.json` en el mismo directorio.

### 16.5 Ejemplo de implementacion de argparse

```python
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos.

    Returns:
        Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - LLM Function Calling"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input"),
        help="Ruta al directorio o archivo de entrada",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Ruta al archivo de salida",
    )
    return parser.parse_args()
```

---

## Capitulo 17: El SDK del LLM - API Completa

### 17.1 Clase Small_LLM_Model

El SDK proporciona una clase `Small_LLM_Model` que envuelve el modelo. Estos
son los metodos que puedes usar:

#### `encode(text: str) -> List[int]`

Convierte texto en una lista de IDs de tokens.

```python
ids = model.encode("Hola mundo")
print(ids)  # [1234, 5678]
```

**Por que lo necesitas**: para convertir el prompt en numeros que el modelo
puede procesar.

#### `decode(token_ids: List[int]) -> str` (opcional)

Convierte una lista de IDs de tokens de vuelta a texto.

```python
text = model.decode([1234, 5678])
print(text)  # "Hola mundo"
```

**Por que lo necesitas**: para depurar y ver que tokens se estan generando.

#### `get_logits_from_input_ids(input_ids: List[int]) -> List[float]`

Dada una secuencia de IDs de tokens, devuelve los logits para el siguiente
token.

```python
logits = model.get_logits_from_input_ids([1234, 5678])
print(len(logits))  # 151643 (tamano del vocabulario)
print(logits[0])     # 0.1 (logit del token 0)
```

**Por que lo necesitas**: es el metodo principal para la decodificacion
restringida. Te da las puntuaciones que necesitas para saber que token
quiere generar el modelo.

#### `get_path_to_vocab_file() -> str`

Devuelve la ruta al archivo `vocab.json` del modelo.

```python
path = model.get_path_to_vocab_file()
print(path)  # "/home/user/.cache/huggingface/hub/.../vocab.json"
```

**Por que lo necesitas**: para cargar el vocabulario y poder mapear entre
IDs y texto.

### 17.2 Flujo completo de uso del SDK

```python
from llm_sdk import Small_LLM_Model

# 1. Cargar el modelo
model = Small_LLM_Model()

# 2. Codificar el prompt
prompt = "What is the sum of 2 and 3?"
input_ids = model.encode(prompt)

# 3. Obtener logits
logits = model.get_logits_from_input_ids(input_ids)

# 4. Elegir el token con mayor logit
best_token_id = max(range(len(logits)), key=lambda i: logits[i])

# 5. Anadir el token al contexto
input_ids.append(best_token_id)

# 6. Decodificar para ver que se genero
generated = model.decode([best_token_id])
print(generated)  # " The" (o lo que sea)
```

### 17.3 Errores comunes con el SDK

| Error | Causa | Solucion |
|-------|-------|----------|
| `prompt_ids + generated` | `encode()` devuelve Tensor | Usa `.flatten().tolist()` |
| `model.get_logits_from_input_ids(tensor)` | El metodo espera una lista | Pasa una lista, no un Tensor |
| Usar `model._model` | Es privado | Usa los metodos publicos |
| Usar `model._tokenizer` | Es privado | Usa `encode()` y `decode()` |

---

## Capitulo 18: Rendimiento y Optimizacion

### 18.1 Requisitos de rendimiento del subject

| Metrica | Requisito |
|--------|-----------|
| Precision | > 95% |
| JSON valido | 100% |
| Tiempo | < 5 minutos para todos los prompts |

### 18.2 Donde se va el tiempo

1. **Carga del modelo**: ~10-30 segundos (solo una vez)
2. **Carga del vocabulario**: ~1 segundo
3. **Generacion de tokens**: ~0.1-0.5 segundos por token
4. **Total para 11 prompts**: ~1-2 minutos

### 18.3 Optimizaciones posibles

#### Cargar el vocabulario una sola vez

```python
# Mal: carga el vocabulario en cada prompt
for prompt in prompts:
    vocab = Vocabulary(model.get_path_to_vocab_file())  # Lento

# Bien: carga el vocabulario una vez
vocab = Vocabulary(model.get_path_to_vocab_file())
for prompt in prompts:
    # usa vocab
```

#### Usar numpy para el enmascaramiento

```python
# Mal: bucle en Python
for i in range(len(logits)):
    if i not in legal:
        logits[i] = -np.inf

# Bien: vectorizado con numpy
masked = np.full(len(logits), -np.inf)
masked[legal] = logits[legal]
```

#### Precalcular el texto de todos los tokens

```python
# Mal: calcular token_text en cada paso
for tid in range(len(vocab)):
    text = vocab.token_text(tid)  # Lento

# Bien: precalcular una vez
id_to_text = [vocab.token_text(i) for i in range(len(vocab))]
```

### 18.4 Como medir el tiempo

```python
import time

start = time.time()
# ... ejecutar el programa ...
end = time.time()
print(f"Tiempo total: {end - start:.2f} segundos")
```

---

## Capitulo 19: Casos de Uso y Ejemplos

### 19.1 Ejemplo basico

**Entrada** (`data/input/function_calling_tests.json`):
```json
["What is the sum of 2 and 3?"]
```

**Entrada** (`data/input/functions_definition.json`):
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

**Ejecucion**:
```bash
uv run python -m src
```

**Salida** (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2.0, "b": 3.0}
  }
]
```

### 19.2 Ejemplo con multiples funciones

**Entrada**:
```json
[
  "Greet shrek",
  "Reverse the string 'hello'",
  "What is the square root of 16?"
]
```

**Salida**:
```json
[
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "fn_name": "fn_reverse_string",
    "args": {"s": "hello"}
  },
  {
    "prompt": "What is the square root of 16?",
    "fn_name": "fn_get_square_root",
    "args": {"a": 16.0}
  }
]
```

### 19.3 Ejemplo con strings complejos

**Entrada**:
```json
["Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"]
```

**Salida**:
```json
[
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "fn_name": "fn_substitute_string_with_regex",
    "args": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

### 19.4 Casos limite a probar

| Caso | Entrada | Comportamiento esperado |
|------|---------|------------------------|
| String vacia | `""` | El programa no debe fallar |
| Numero muy grande | `"What is the sum of 999999999 and 1?"` | Debe funcionar |
| Caracteres especiales | `"Greet O'Brien"` | Debe manejar el apostrofo |
| Prompt ambiguo | `"What is the result?"` | El modelo elige una funcion |
| JSON invalido en entrada | Archivo con sintaxis incorrecta | Mensaje de error claro |
| Archivo inexistente | `data/input/no_existe.json` | Mensaje de error claro |

---

## Capitulo 20: Estrategia de Pruebas con la Moulinette

### 20.1 Que es la moulinette

La moulinette es el programa de correccion. Tiene dos modos:

1. **`prepare_exercises`**: genera los archivos de entrada (tests y
   correcciones) para un conjunto de funciones (publicas o privadas).
2. **`grade_student_answers`**: compara tu salida con la esperada.

### 20.2 Como preparar los ejercicios

```bash
cd moulinette

# Generar ejercicios publicos
uv run python -m moulinette prepare_exercises --set public

# Generar ejercicios privados
uv run python -m moulinette prepare_exercises --set private
```

Esto crea:
- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`
- `data/correction/function_calling_corrections.json`

### 20.3 Como corregir

```bash
# Primero ejecuta tu programa
cd ..
uv run python -m src

# Luego corrige
cd moulinette
uv run python -m moulinette grade_student_answers ../data/output/function_calling_results.json
```

### 20.4 Que verifica la moulinette

| Check | Detalle |
|-------|---------|
| Prompt | Que el prompt coincide con el esperado |
| fn_name | Que el nombre de la funcion es correcto |
| Argumentos | Que los argumentos son correctos (nombre y tipo) |
| Resultado | Que llamar la funcion con esos argumentos da el resultado esperado |

### 20.5 Interpretar los resultados

```
Test 1/11
Prompt: What is the sum of 2 and 3?
>>> VALID <<<
```

- **VALID**: tu respuesta es correcta.
- **INVALID: prompt mismatch**: el prompt no coincide.
- **INVALID: unknown function**: elegiste una funcion que no existe.
- **INVALID: invalid parameters**: los argumentos no son validos.
- **INVALID: wrong output**: la funcion con esos argumentos no da el
  resultado esperado.

---

## Capitulo 21: Errores Especificos de este Proyecto

### 21.1 El problema de los nombres como prefijos

Si tienes dos funciones: `fn_add` y `fn_add_numbers`, cuando la maquina esta
en fase NAME y ha generado `fn_add`, no sabe si el nombre completo es
`fn_add` o `fn_add_numbers`.

**Solucion**: asumir que ningun nombre es prefijo de otro (como hace el
repo de referencia). Si lo son, habria que esperar la comilla de cierre para
desambiguar.

### 21.2 El problema de los numeros negativos

Un numero negativo empieza con `-`. Pero `-` también podria ser parte de un
token que no es un numero. La maquina debe saber que en fase VALUE_NUMBER,
el `-` solo es valido al principio.

### 21.3 El problema de los strings con comillas

Si un valor string contiene `"`, debe escaparse como `\"`. La maquina debe
rastrear si el caracter anterior era `\` para saber si una `"` cierra la
string o es parte del contenido.

### 21.4 El problema de los strings con caracteres no imprimibles

Si un token contiene caracteres no imprimibles (como tabulaciones o saltos
de linea), la maquina debe decidir si son validos en una string JSON. En
general, JSON no permite caracteres de control sin escapar, pero para este
proyecto puedes aceptar cualquier caracter imprimible.

### 21.5 El problema del token vacio

Algunos tokens pueden decodificar a un string vacio (por ejemplo, tokens
especiales). La maquina debe rechazar estos tokens en `accepts()`.

---

## Capitulo 22: Preguntas Frecuentes

### P: Puedo usar torch en src/?

**R**: No. El subject lo prohibe. Usa solo `llm_sdk`.

### P: Puedo usar numpy?

**R**: Si. El subject lo permite explicitamente.

### P: Tengo que usar Qwen3-0.6B?

**R**: Si, por defecto. Puedes usar otros modelos si funciona con
Qwen3-0.6B.

### P: Como elige el modelo que funcion llamar?

**R**: El modelo genera logits para todos los tokens. Tu codigo filtra
solo los tokens que corresponden a nombres de funcion validos. El token
con mayor logit entre esos es el elegido.

### P: Que pasa si el modelo quiere generar una funcion que no existe?

**R**: No puede. La maquina de estados solo permite caracteres que
continuan un nombre de funcion valido. Si el modelo "quiere" generar
`fn_unknown`, la maquina no le dejara.

### P: Como se que mi JSON es 100% valido?

**R**: Por construccion. La maquina de estados solo permite caracteres
que mantienen el JSON valido. Si el JSON final es invalido, hay un bug
en la maquina.

### P: Puedo hacer que el programa sea mas rapido?

**R**: Si. Las optimizaciones principales son:
- Cargar el vocabulario una sola vez.
- Usar numpy para el enmascaramiento.
- Precalcular el texto de todos los tokens.

### P: Que hago si la moulinette dice que mi salida es invalida?

**R**: Revisa:
1. Que el prompt coincide exactamente.
2. Que el nombre de la funcion es correcto.
3. Que los argumentos tienen los tipos correctos (number vs string).
4. Que no hay claves extra en el JSON.

---

## Capitulo 23: Resumen del Orden de Trabajo (Actualizado)

1. **Entender el subject**: lee el PDF y anota los requisitos.
2. **Preparar entorno**: `uv sync`.
3. **Estudiar conceptos**: tokenizacion, logits, decodificacion restringida.
4. **Implementar `models.py`**: validacion con pydantic.
5. **Implementar `vocabulary.py`**: mapeo token <-> texto.
6. **Implementar `constraints.py`**: la maquina de estados.
7. **Implementar `decoder.py`**: el bucle de decodificacion.
8. **Implementar `pipeline.py`**: union de todo.
9. **Implementar `io_utils.py` y `__main__.py`**: entrada/salida y CLI.
10. **Makefile**: todos los targets obligatorios.
11. **Tests**: pytest para constraints y decoder.
12. **README.md**: todas las secciones obligatorias.
13. **Validar**: lint, tests, ejecucion real, moulinette.

---

## Capitulo 24: Referencias y Enlaces

### Documentacion oficial

- **uv**: https://docs.astral.sh/uv/
- **Pydantic**: https://docs.pydantic.dev/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Tokenizer summary**: https://huggingface.co/docs/transformers/en/tokenizer_summary
- **Qwen3-0.6B**: https://huggingface.co/Qwen/Qwen3-0.6B

### Articulos y tutoriales

- **Decoding strategies**: https://lilianweng.github.io/posts/2023-01-27-decoding/
- **GPT-2 encoder (byte BPE)**: https://github.com/openai/gpt-2/blob/master/src/encoder.py
- **Softmax function**: https://en.wikipedia.org/wiki/Softmax_function

### Repositorios de referencia

- **Referencia 1**: https://github.com/eepylaurie/42-call_me_maybe
  (implementacion completa con tests, CI, README detallado)
- **Referencia 2**: https://github.com/ayfadli/Call-Me-Maybe
  (otra aproximacion, mas simple)

### Conceptos clave para buscar

- "Constrained decoding LLM"
- "Byte-Pair Encoding BPE"
- "Logits and softmax"
- "JSON schema validation"
- "State machine pattern Python"
- "Pydantic BaseModel"
