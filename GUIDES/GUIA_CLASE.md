# GUÍA UNIVERSITARIA: Call Me Maybe

## Una clase paso a paso para entender y construir el proyecto

**Duración estimada:** 4-5 horas

---

## Índice

**PARTE I — Los fundamentos**
- Lección 1: El problema que vamos a resolver
- Lección 2: Cómo funciona un modelo de lenguaje por dentro
- Lección 3: La decodificación restringida

**PARTE II — Manos a la obra**
- Lección 4: Preparar el entorno de trabajo
- Lección 5: Crear los modelos de datos (`models.py`)
- Lección 6: Crear las herramientas de entrada/salida (`tools.py`)
- Lección 7: Crear el constructor de prompts (`prompt_builder.py`)
- Lección 8: Crear el vocabulario (`vocab.py`)
- Lección 9: Crear el decodificador restringido (`constrained_decoder.py`)
- Lección 10: Crear el orquestador (`function_caller.py`)
- Lección 11: Crear el punto de entrada (`__main__.py`)
- Lección 12: Ejecutar y depurar el programa

**Anexo**
- Glosario

---

# PARTE I: LOS FUNDAMENTOS (60 minutos)

Antes de escribir una sola línea de código, necesitas entender tres conceptos fundamentales. Sin ellos, el código no tendrá sentido.

---

## LECCIÓN 1: El problema que vamos a resolver

### 1.1. El mundo antes de este proyecto

Imagina que trabajas en una empresa y te piden crear un sistema que entienda lenguaje humano. Por ejemplo, un usuario escribe:

> "¿Cuánto es 40 más 2?"

Y tu sistema debe ser capaz de:
1. Entender que quiere sumar
2. Identificar los números 40 y 2
3. Llamar a una función de suma con esos valores

Eso suena fácil, pero hay un problema fundamental: **los ordenadores no entienden el lenguaje humano**. Solo entienden instrucciones muy precisas y estructuradas.

### 1.2. La solución obvia (y por qué no funciona)

La primera idea que se te podría ocurrir es usar un modelo de lenguaje como ChatGPT. Le preguntas "¿qué función debo llamar?" y te responde.

Pero aquí viene el problema: **los modelos de lenguaje son torpes con el formato**. Si le pides a un modelo que escriba JSON (un formato de datos muy estructurado), fallará muchas veces.

¿Por qué? Porque los modelos de lenguaje están diseñados para generar texto natural, no para seguir reglas estrictas de formato. Es como pedirle a un poeta que escriba un formulario de Hacienda: puede que lo haga bien, pero probablemente se invente cosas o se salte campos.

### 1.3. El problema específico de los modelos pequeños

En este proyecto no vamos a usar un modelo enorme como GPT-4. Vamos a usar un modelo pequeño llamado Qwen3-0.6B, que tiene solo 600 millones de parámetros.

Para que te hagas una idea:
- GPT-4 tiene más de 1 billón de parámetros
- Nuestro modelo tiene 600 millones
- Es como comparar un autobús con una bicicleta

Los modelos pequeños son mucho más torpes con el formato. Si le pides a Qwen3-0.6B que genere JSON directamente, fallará con mucha frecuencia. Generará cosas como:

```
{"function": "sumar"}   ← le falta el campo "argumentos"
```

O peor:

```
La función es sumar con a=40 y b=2   ← esto no es JSON
```

### 1.4. La solución: no dejar que se equivoque

La idea clave del proyecto es: **no le preguntes al modelo qué quiere escribir. Oblígalo a escribir solo lo que es válido.**

En lugar de pedirle al modelo "escribe un JSON con la función y los argumentos", vamos a hacer algo mucho más inteligente:

1. Le preguntamos al modelo qué token (palabra o parte de palabra) quiere escribir a continuación
2. Antes de que elija, le decimos: "solo puedes elegir entre estos 3 tokens que son válidos"
3. El modelo elige el que más le guste de esos 3
4. Repetimos el proceso para el siguiente token

Así es **físicamente imposible** que el modelo escriba algo incorrecto. Es como si a un niño le das a elegir entre "manzana", "pera" y "plátano" — puede elegir cualquiera, pero nunca va a elegir "coche" porque no está entre las opciones.

Esta técnica se llama **decodificación restringida** (constrained decoding en inglés) y es el corazón del proyecto.

---

## LECCIÓN 2: Cómo funciona un modelo de lenguaje por dentro

### 2.1. Los tokens: los ladrillos del lenguaje

Antes de entender la decodificación restringida, necesitas entender cómo funciona un modelo de lenguaje por dentro.

Los modelos NO entienden letras. Tampoco entienden palabras exactamente. Entienden **tokens**.

Un token es un trozo de texto. Puede ser:
- Una palabra entera: "hola", "mundo", "42"
- Parte de una palabra: "in", "creíble" (para "increíble")
- Un signo: "{", "}", ":"
- Un espacio: " hola" (sí, el espacio cuenta como parte del token)

El modelo Qwen3-0.6B tiene un vocabulario de aproximadamente **150.000 tokens diferentes**. Cada token tiene un número de identificación (ID) único.

Piensa en el vocabulario como un diccionario gigante:
- La palabra "hola" tiene el ID 1284
- La palabra "mundo" tiene el ID 339
- El signo "{" tiene el ID 42
- El signo "}" tiene el ID 43

### 2.2. Input IDs: el modelo solo entiende números

Cuando le pasas texto al modelo, primero lo conviertes a números (IDs de tokens). Este proceso se llama **tokenización**.

Por ejemplo, si le pasas "Hola mundo", el modelo no ve las letras H-o-l-a-m-u-n-d-o. Ve una lista de números: [1284, 339].

Esa lista de números es lo que el modelo recibe. Se llama **input_ids**.

### 2.3. Logits: las puntuaciones del modelo

Cuando el modelo recibe los IDs, no te devuelve directamente "la siguiente palabra es X". Te devuelve una **puntuación** para cada uno de los 150.000 tokens de su vocabulario.

Esa puntuación se llama **logit**. Es un número que puede ser positivo o negativo. Cuanto más alto, más probable cree el modelo que es ese token.

Imagínate que el modelo recibe "¿Cuánto es 2+3?" y devuelve algo así (simplificado a 5 tokens en vez de 150.000):

- Token "the" → logit = 0.3
- Token " is" → logit = 2.1
- Token "?" → logit = 5.4 (el más alto)
- Token "sum" → logit = 1.8
- Token "{" → logit = 0.1

El modelo cree que lo más probable después de "¿Cuánto es 2+3?" es "?". Tiene sentido: es una pregunta.

Normalmente, la IA elegiría el token con el logit más alto (el 5.4, que es "?"). Pero nosotros vamos a **modificar** esas puntuaciones.

### 2.4. Softmax: convertir puntuaciones en probabilidades

Los logits son números en bruto, difíciles de interpretar. Para convertirlos en probabilidades (números entre 0 y 1 que suman 1), se usa **softmax**.

La fórmula es: softmax(x_i) = e^(x_i) / suma(e^(x_j))

No te asustes con la fórmula. Lo que hace es:
1. Coge cada logit
2. Calcula el número e^logit (e ≈ 2.71828 elevado al logit)
3. Divide cada uno por la suma de todos

El resultado: probabilidades que suman 1. Un token con probabilidad 0.8 significa que el modelo cree que hay un 80% de probabilidades de que sea ese.

En el proyecto usamos **log-softmax**, que es el logaritmo de la softmax. Es más estable numéricamente y más práctico para sumar probabilidades (en vez de multiplicarlas).

### 2.5. Argmax: elegir el mejor

El argmax es una operación que elige el elemento con valor máximo. Si tienes [0.3, 2.1, 5.4, 0.1], el argmax devuelve el índice 2 (el token con puntuación 5.4).

---

## LECCIÓN 3: La decodificación restringida

### 3.1. ¿Qué es?

La decodificación restringida es una técnica que filtra las opciones del modelo en cada paso para que solo pueda elegir tokens válidos.

### 3.2. Analogía: el menú del restaurante

Imagina que vas a un restaurante con un menú de 150 platos. El camarero (el modelo) te pregunta qué quieres.

**Sin restricción:** El camarero puede recomendarte cualquier plato, incluso los que no están en el menú.

**Con restricción:** Tú le dices "solo puedes elegir entre paella, sushi o pizza". Ahora es imposible que te recomiende una hamburguesa.

En el proyecto:
- El "menú" son los 150.000 tokens del vocabulario del modelo
- La "restricción" es: "solo puedes elegir tokens que mantengan el JSON válido"

### 3.3. ¿Cómo funciona paso a paso?

Paso 1: El modelo genera puntuaciones para todos los tokens
- Token "{" → puntuación 5.2
- Token "}" → puntuación 3.1
- Token "a" → puntuación 0.5
- Token "z" → puntuación 0.1
- ... (150.000 puntuaciones)

Paso 2: Tú pones a -infinito los tokens que NO quieres
- Token "{" → 5.2 (válido, se queda)
- Token "}" → -infinito (inválido, aún no toca cerrar)
- Token "a" → -infinito (inválido)
- Token "z" → -infinito (inválido)

Paso 3: El modelo solo puede elegir entre los tokens con puntuación real
- Elige "{" (el único válido)

Paso 4: Repites el proceso para el siguiente token

### 3.4. ¿Por qué -infinito y no 0?

Porque el modelo elige el token con el logit MÁS ALTO. Si pusiéramos 0, algunos tokens prohibidos podrían tener logits negativos (por ejemplo, -5) y el 0 sería mayor que esos, con lo que el token prohibido sería elegido.

Con -infinito, ningún token prohibido puede ser elegido jamás. Es la puntuación más baja posible.

---

### 3.5. Qué garantiza la decodificación restringida (y qué no)

Conviene tenerlo claro desde el principio, porque es la duda más habitual. La decodificación restringida **no hace al modelo más listo**: solo le prohíbe salirse del formato. Hay dos responsabilidades distintas:

| Responsabilidad | ¿Quién? | ¿Garantizado? |
|---|---|---|
| Estructura JSON válida (llaves, comas, comillas) | El decoder (nuestro código) | Sí |
| Claves exactamente `prompt`, `fn_name`, `args` | El decoder | Sí |
| Que `fn_name` sea una de las funciones disponibles | El decoder | Sí |
| Que el valor respete el tipo (`number`/`string`/`boolean`) | El decoder | Sí |
| **Qué** función elegir | El modelo | No |
| **Qué** valor poner (p. ej. una regex concreta) | El modelo | No |

**Ejemplo típico de confusión.** Si el argumento `regex` es de tipo string, el decoder garantiza que el valor será un string bien formado dentro del JSON. Pero *qué* string elige el modelo (`\d+`, `[0-9]+`, `34|233`...) es decisión suya. Todos son JSON válido; que acierte con la regex "correcta" depende de la capacidad del modelo, no del decoder.

**Consecuencia práctica:** si el JSON se parsea pero el valor no es el esperado, no se arregla "forzando más el formato"; habría que cambiar de modelo o de prompt.

> **Matiz.** El decoder elimina los caracteres estructurales que romperían el JSON, pero no valida la sintaxis interna de un valor. Por ejemplo, la máscara de `number` permite dígitos, `.` y `-`, así que en teoría podría salir un número mal formado como `1.2.3`. Con los casos del subject no ocurre; aun así, es la parte que el decoder *no* garantiza.

---

# PARTE II: MANOS A LA OBRA (3-4 horas)

Ahora que entiendes la teoría, vamos a construir el proyecto paso a paso. Cada lección se centra en una función o documento específico.

---

## LECCIÓN 4: Preparar el entorno de trabajo

### 4.1. Crear la estructura de carpetas

**Qué hacer:**
Crea las carpetas que necesitas para el proyecto.

**Cómo hacerlo:**
Abre tu terminal y ejecuta:
```bash
mkdir -p src llm_sdk data/input data/output GUIDES
```

**Por qué lo haces:**
Necesitas una estructura organizada para que el programa pueda encontrar los archivos. La carpeta `src/` contendrá tu código, `data/input/` los archivos de entrada, `data/output/` los resultados, y `llm_sdk/` el kit de herramientas del modelo.

**Cómo debería funcionar:**
Después de ejecutar el comando, deberías ver las carpetas creadas. Puedes comprobarlo con `ls -la`.

---

### 4.2. Crear pyproject.toml

**Qué hacer:**
Crea un archivo `pyproject.toml` en la raíz del proyecto con las dependencias necesarias.

**Cómo hacerlo:**
Crea un archivo llamado `pyproject.toml` con este contenido:

```toml
[project]
name = "call-me-maybe"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pydantic",
]

[tool.mypy]
strict = true
```

**Por qué lo haces:**
Este archivo le dice a Python qué dependencias necesita el proyecto (numpy para números, pydantic para validación de datos). La sección `[tool.mypy]` activa el modo estricto de verificación de tipos.

**Cómo debería funcionar:**
Cuando ejecutes `uv sync`, Python leerá este archivo e instalará las dependencias automáticamente.

---

### 4.3. Crear el Makefile

**Qué hacer:**
Crea un archivo `Makefile` con atajos para comandos frecuentes.

**Cómo hacerlo:**
Crea un archivo llamado `Makefile` con este contenido:

```makefile
install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

lint:
	uv run flake8 src
	uv run mypy src
```

**Por qué lo haces:**
El Makefile te permite escribir `make run` en vez de `uv run python -m src --input data/input --output data/output/function_calling_results.json`. Es más rápido y menos propenso a errores.

**Cómo debería funcionar:**
Cuando escribas `make run`, el programa debería ejecutarse. Si hay errores, el Makefile te mostrará mensajes de error claros.

---

### 4.4. Copiar llm_sdk

**Qué hacer:**
Copia la carpeta `llm_sdk` en la raíz del proyecto.

**Cómo hacerlo:**
Si tienes el SDK en otro lugar, cópialo:
```bash
cp -r /ruta/al/llm_sdk .
```

**Por qué lo haces:**
El SDK contiene la clase `Small_LLM_Model` que te permite interactuar con el modelo Qwen3-0.6B sin instalar pytorch ni transformers.

**Cómo debería funcionar:**
Después de copiarlo, deberías ver la carpeta `llm_sdk/` con un archivo `__init__.py` dentro.

---

### 4.5. Crear los archivos de datos de entrada

**Qué hacer:**
Crea los archivos JSON con las funciones disponibles y las preguntas de prueba.

**Cómo hacerlo:**
Crea `data/input/functions_definition.json` con las funciones que el sistema puede usar. Cada función tiene un nombre, una descripción, parámetros y un tipo de retorno.

Crea `data/input/function_calling_tests.json` con las preguntas que el sistema debe procesar.

**Por qué lo haces:**
Estos archivos son la entrada del programa. Sin ellos, el programa no tiene nada que procesar.

**Cómo debería funcionar:**
Cuando el programa se ejecute, leerá estos archivos y procesará cada pregunta.

---

## LECCIÓN 5: Crear los modelos de datos (models.py)

### 5.1. ¿Qué son los modelos de datos y por qué existen?

Los modelos de datos son "plantillas" que definen cómo deben ser los datos. Si un dato no encaja en la plantilla, el programa lanza un error.

**Por qué los necesitas:**
Imagina que el archivo de funciones tiene un error: un parámetro que debería ser un texto pero es un número. Sin validación, el programa fallaría más tarde de forma misteriosa. Con Pydantic, el error se detecta inmediatamente.

**Por qué se llaman "modelos":**
Se llaman "modelos" porque son como un modelo o molde que define la estructura de los datos. En programación, un "modelo" es una representación abstracta de algo. En este caso, el modelo representa cómo debe ser una función, un parámetro o un resultado.

**Cómo funciona BaseModel:**
BaseModel es la clase base de Pydantic. Cuando heredas de BaseModel, estás diciendo a Python: "esta clase es un modelo de datos". BaseModel proporciona automáticamente:
- Validación de tipos: si dices que un campo debe ser un string, BaseModel verifica que sea un string
- Conversión de datos: si pasas un número como string, BaseModel intenta convertirlo
- Errores claros: si algo está mal, BaseModel lanza un error que te dice exactamente qué está mal
- Serialización: puedes convertir el modelo a JSON fácilmente

**Analogía:**
Es la misma idea que el "formulario de Hacienda" de la Lección 1: campos con reglas que se validan solas.

---

### 5.2. Crear ParameterDefinition

**Qué hacer:**
Un único campo `type: str` para cada parámetro (por ejemplo, `"number"`, `"string"` o `"boolean"`).

**Qué debe hacer:**
Que Pydantic valide que `type` sea texto; si no, lanzará un error claro.

---

### 5.3. Crear FunctionDefinition

**Qué hacer:**
Cuatro campos: `name: str`, `description: str`, `parameters: dict[str, ParameterDefinition]` y `returns: dict[str, str]`.

**Qué debe hacer:**
Validar que todos los campos estén y sean del tipo correcto, y convertir automáticamente cada diccionario de `parameters` en un `ParameterDefinition`.

---

### 5.4. Crear FunctionCall

**Qué hacer:**
Tres campos: `prompt: str` (la pregunta original), `fn_name: str` (la función elegida) y `args: dict[str, object]` (los argumentos extraídos).

**Qué debe hacer:**
Validar que los tres estén presentes y sean del tipo correcto. Este objeto es el resultado final del programa.

---

### 5.5. Crear TestPrompt

**Qué hacer:**
Un único campo `prompt: str`.

**Qué debe hacer:**
Validar que el prompt sea texto. Representa cada pregunta de prueba que el sistema debe procesar.

---

### 5.6. Crear JSONState (los estados de la máquina)

**Qué hacer:**
Define un `Enum` con los estados que recorrerá el decodificador al construir el JSON.

**Qué necesita:**
- Un miembro por cada tramo del JSON (nombre, apertura de `args`, clave, valor, cierre...)

**Qué debe hacer:**
- Heredar de `Enum`
- Asignar un valor a cada estado (basta con `auto()`)

```python
from enum import Enum, auto


class JSONState(Enum):
    FN_NAME = auto()       # Forzado: nombre de la función
    AFTER_NAME = auto()    # Forzado: ", "args": {
    PARAM_KEY = auto()     # Forzado: "param":
    PARAM_VALUE = auto()   # El modelo elige el valor
    NEXT_PARAM = auto()    # Forzado: , "siguiente": o el cierre
    END = auto()           # Forzado: }
    FINISHED = auto()      # Fin
```

**Por qué lo haces:**
En lugar de llevar el progreso con un booleano o un string "mágico", usas estados con nombre. Así el decodificador sabe en cada momento qué tokens son válidos (ver Lección 9.6).

**Cómo debería funcionar:**
Cada iteración del bucle de generación está en un estado; según el estado se emiten unos tokens u otros y se pasa al siguiente, hasta llegar a `FINISHED`.

> En `models.py` también encontrarás un `ParameterType` (`Enum` con `number` y `string`) para nombrar los tipos. El decodificador, en cambio, compara con los strings `"number"`, `"string"` y `"boolean"`.

---

## LECCIÓN 6: Crear las herramientas de entrada/salida (tools.py)

### 6.1. Implementar json_reader

**Qué es json_reader?**

Es una función que lee un archivo JSON y lo convierte en datos Python. Es la base de todo: sin ella, el programa no puede leer los archivos de entrada.

**Qué necesita hacer:**

1. **Recibir una ruta de archivo** (un string o un objeto Path)
2. **Verificar que el archivo exista** — Si no existe, lanzar un error claro
3. **Abrir el archivo** — Usa un context manager (`with open(...)`) para asegurar que el archivo se cierra automáticamente
4. **Cargar el JSON** — Usa la librería `json` para convertir el contenido del archivo en datos Python
5. **Devolver los datos** — El resultado puede ser un diccionario, una lista, o cualquier estructura JSON

**Por qué es importante:**

- Sin esta función, el programa no puede leer las funciones disponibles ni las preguntas de prueba
- Es la base para todas las demás funciones de lectura/escritura
- Debe manejar errores claramente: si el archivo no existe o el JSON está mal formado, el usuario debe entender qué pasó

**Cómo debería funcionar:**

- Si el archivo existe y el JSON es válido → devuelve los datos
- Si el archivo no existe → lanza un error claro ("Archivo no encontrado: {ruta}")
- Si el JSON está mal formado → `json.load` lanza `JSONDecodeError` (puedes capturarlo para dar un mensaje más claro)

**Conceptos clave:**

#### Context manager
Un context manager es una forma de asegurar que un recurso se cierra automáticamente cuando terminas de usarlo. En Python, se usa la palabra clave `with`:

```python
with open("archivo.txt", "r") as f:
    # Aquí el archivo está abierto
    contenido = f.read()
# Aquí el archivo se ha cerrado automáticamente
```

#### Manejo de errores
Es importante manejar errores claramente para que el usuario entienda qué pasó. Usa try/except para capturar errores y lanzar mensajes claros.

---

### 6.2. Implementar load_function_def

**Qué es load_function_def?**

Es una función que carga las definiciones de funciones desde un archivo JSON y las valida con Pydantic. Convierte los datos JSON en objetos `FunctionDefinition`.

**Qué necesita hacer:**

1. **Recibir una ruta de archivo** — El archivo JSON con las definiciones de funciones
2. **Leer el archivo** — Usa `json_reader` (la función que acabas de crear)
3. **Verificar que sea una lista** — El JSON debe ser un array de funciones
4. **Validar cada función** — Para cada elemento de la lista, crear un objeto `FunctionDefinition`
5. **Manejar errores** — Si alguna definición es inválida, indicar en qué índice falló
6. **Devolver la lista de funciones** — Una lista de objetos `FunctionDefinition`

**Por qué es importante:**

- Sin esta función, el programa no puede usar las funciones disponibles
- Valida que las funciones tengan el formato correcto antes de usarlas
- Si hay un error en el JSON, el usuario debe entender exactamente qué está mal y dónde

**Cómo debería funcionar:**

- Si el JSON es válido → devuelve una lista de objetos `FunctionDefinition`
- Si el JSON no es una lista → lanza un error claro
- Si alguna función es inválida → lanza un error indicando el índice

**Conceptos clave:**

#### Verificar tipos
Usa `isinstance(data, list)` para verificar que los datos son una lista antes de iterar.

#### Iterar y crear objetos
Para cada elemento en la lista, crea un objeto `FunctionDefinition` y añádelo a una lista.

#### Manejar errores en bucles
Usa try/except dentro del bucle para capturar errores e indicar qué elemento falló.

---

### 6.3. Implementar load_prompt

**Qué es load_prompt?**

Es una función que carga los prompts de prueba desde un archivo JSON. Los prompts son las preguntas que el sistema debe procesar.

**Qué necesita hacer:**

1. **Recibir una ruta de archivo** — El archivo JSON con los prompts
2. **Leer el archivo** — Usa `json_reader`
3. **Verificar que sea una lista** — El JSON debe ser un array de prompts
4. **Extraer el texto de cada prompt** — Cada prompt puede ser un string o un diccionario con un campo "prompt"
5. **Devolver una lista de strings** — Una lista con los textos de los prompts

**Por qué es importante:**

- Sin esta función, el programa no puede leer las preguntas de prueba
- Los prompts pueden venir en diferentes formatos (string o diccionario), así que la función debe manejar ambos casos
- Es la última función de lectura que necesitas antes de empezar con la lógica del modelo

**Cómo debería funcionar:**

- Si el JSON es válido → devuelve una lista de strings
- Si el JSON no es una lista → lanza un error claro
- Si un prompt es un diccionario → extrae el campo "prompt"
- Si un prompt es un string → lo usa directamente

**Conceptos clave:**

#### Manejar múltiples formatos
Los prompts pueden venir como strings o diccionarios. Usa `isinstance()` para verificar el tipo y manejar cada caso.

#### TestPrompt
Es una clase Pydantic que representa un prompt de prueba. Tiene un campo `prompt` de tipo string.

---

### 6.4. Implementar json_exporter

**Qué es json_exporter?**

Es una función que escribe los resultados en un archivo JSON. Es la última función de `tools.py`.

**Qué necesita hacer:**

1. **Recibir una lista de objetos `FunctionCall`** — Los resultados del programa
2. **Recibir una ruta de archivo** — Dónde guardar el JSON
3. **Crear la carpeta si no existe** — Si la carpeta `data/output/` no existe, la crea
4. **Convertir los objetos a diccionarios** — Los objetos `FunctionCall` no se pueden escribir directamente a JSON, hay que convertirlos
5. **Escribir el JSON en el archivo** — Usa un context manager para abrir el archivo y escribir

**Por qué es importante:**

- Sin esta función, el programa no puede guardar los resultados
- Debe crear la carpeta si no existe, porque si no, fallará al intentar escribir
- Los objetos Pydantic tienen un método para convertirse a diccionarios

**Cómo debería funcionar:**

- Si la carpeta no existe → la crea
- Si el archivo se escribe correctamente → no devuelve nada (o devuelve None)
- Si hay un error → lanza un error claro

**Conceptos clave:**

#### pathlib
`pathlib` es una librería estándar de Python para trabajar con rutas de archivos. Está permitida en el subject porque es parte de la stdlib.

#### Crear carpetas
Usa `path.mkdir(parents=True, exist_ok=True)` para crear una carpeta si no existe. El parámetro `parents=True` crea las carpetas padre si no existen, y `exist_ok=True` evita errores si la carpeta ya existe.

#### Convertir objetos Pydantic a diccionarios
Los objetos Pydantic tienen un método `.dict()` que los convierte en diccionarios. Esto es necesario porque JSON no puede serializar objetos Pydantic directamente.

#### Escribir JSON con formato legible
Usa `json.dump(data, f, indent=4)` para escribir JSON con sangría de 4 espacios. Esto hace que el JSON sea más legible para los humanos.

---

## LECCIÓN 7: Crear el constructor de prompts (prompt_builder.py)

### 7.1. ¿Qué es un prompt?

Un prompt es el texto que le pasamos al modelo para guiar su respuesta.

**Por qué lo necesitas:**
La forma en que le preguntamos al modelo afecta mucho a la respuesta. Un buen prompt hace que el modelo entienda exactamente qué quieres.

---

### 7.2. Implementar fun_selection

**Qué hace:**
Construye un texto que pregunta al modelo qué función quiere usar.

**Qué necesita:**
1. La pregunta del usuario (ej: "What is the sum of 2 and 3?")
2. La lista de funciones disponibles (ej: fn_add_numbers, fn_greet, etc.)

**Qué debe hacer:**
1. **Empieza con un rol** — Dile al modelo quién es:
   ```
   "You are a function caller assistant"
   ```

2. **Explica qué debe hacer** — Dile al modelo qué tarea tiene:
   ```
   "Select the most appropriate function for the user's request."
   ```

3. **Lista las funciones disponibles** — Una línea por función (nombre y descripción), unidas por saltos de línea:
   ```
   "- fn_add_numbers: Add two numbers together and return their sum.
   - fn_greet: Generate a greeting message for a person by name."
   ```
   (En el código se antepone el texto `" The available functions are: "`.)

4. **Incluye la pregunta del usuario** — Dile al modelo qué pregunta quiere responder:
   ```
   "What is the sum of 2 and 3?"
   ```

5. **Termina con una indicación** — Dile al modelo qué debe generar:
   ```
   "Function to call: "
   ```

**El resultado final sería:**
```
You are a function caller assistant
Select the most appropriate function for the user's request.
 The available functions are: - fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
What is the sum of 2 and 3?
Function to call: 
```

> Fíjate en que el texto se construye **concatenando** trozos: `fun_selection` une las descripciones con `"\n".join(...)`. Por eso "The available functions are:" queda pegado a la primera función.

**¿Por qué termina con "Function to call: "?**
Porque el modelo "quiere" continuar con el nombre de una función. Es como si le diéramos la primera palabra de una frase y esperáramos que la complete.

**Conceptos clave:**

#### Acceder a atributos de objetos
Un objeto `FunctionDefinition` tiene atributos: `name`, `description`, `parameters`, `returns`. Para acceder a ellos, usas el punto:
```python
function.name        # El nombre de la función
function.description # La descripción de la función
```

#### f-strings
Los f-strings son strings que pueden incluir variables:
```python
nombre = "fn_add_numbers"
texto = f"- {nombre}: Add two numbers together"
```

#### Unir strings con saltos de línea
Usa `"\n".join(lista)` para unir todos los strings de una lista con saltos de línea:
```python
funciones_texto = "\n".join(formated_func)
```

#### No es hardcodeado
El modelo es el que decide qué función llamar, no nosotros. La decodificación restringida (que implementaremos después) es lo que garantiza que el modelo solo pueda elegir entre las funciones disponibles.

---

## LECCIÓN 8: Crear el vocabulario (vocab.py)

### 8.1. ¿Qué es un vocabulario?

Un **vocabulario** es el mapeo entre tokens y sus IDs (ya viste en la Lección 2 qué es un token y un ID).

**¿Por qué lo necesitamos?**
El modelo solo entiende números, pero nosotros queremos saber **qué palabra** corresponde a cada número y al revés. Para eso necesitamos el vocabulario.

**¿Cómo se usa en la decodificación restringida?**
Dado un conjunto de caracteres permitidos, buscamos en el vocabulario todos los tokens formados por esos caracteres y obtenemos sus IDs. Después bloqueamos todos los demás. Por ejemplo, para un número buscamos los tokens que son solo dígitos:

```
"0" → ID 10
"1" → ID 11
"2" → ID 12
...
```

Con esos IDs, el modelo solo podrá elegir entre ellos (ver Lección 9).

---

### 8.2. Cargar el archivo vocab.json

**Qué hacer:**
Carga el archivo vocab.json del modelo y construye diccionarios de búsqueda.

**Qué necesita:**
- La ruta al archivo vocab.json del modelo

**Qué debe hacer:**
- Cargar el archivo JSON
- Detectar automáticamente el formato del archivo (clave = token o clave = ID)
- Construir dos diccionarios: uno para buscar IDs a partir de tokens y otro para buscar tokens a partir de IDs

**Por qué lo haces:**
El archivo vocab.json puede tener 150.000 entradas. Necesitas una forma eficiente de buscar tokens e IDs.

**Cómo debería funcionar:**
Cuando creas un objeto VocabIndex, carga el archivo y construye los diccionarios. Después, podrás buscar tokens e IDs rápidamente.

**Formato del archivo:**

El `vocab.json` del modelo es un diccionario JSON. Puede tener dos formatos:

**Formato A** (clave = token, valor = ID):
```json
{"hola": 1284, "mundo": 339, "{": 42}
```

**Formato B** (clave = ID, valor = token):
```json
{"1284": "hola", "339": "mundo", "42": "{"}
```

**Detectar el formato automáticamente:**

Para detectar el formato, puedes mirar la primera clave del diccionario:
- Si la primera clave es un número → Formato B
- Si la primera clave es un texto → Formato A

Usa try/except para intentar convertir la primera clave a número:

```python
try:
    int(first_key)
    # Es un número → Formato B (clave = ID, valor = token)
except (ValueError, TypeError):
    # No es un número → Formato A (clave = token, valor = ID)
```

**Crear diccionarios de búsqueda:**

Necesitas dos diccionarios:
1. **token → ID** — Para buscar el ID de un token
2. **ID → token** — Para buscar el token de un ID

**Si es Formato A** (`{"token": id}`):
```python
self.token_to_id = data  # {"hola": 1284, "mundo": 339}
self.id_to_token = {v: k for k, v in data.items()}  # {1284: "hola", 339: "mundo"}
```

**Si es Formato B** (`{"id": token}`):
```python
self.id_to_token = data  # {"1284": "hola", "339": "mundo"}
self.token_to_id = {v: k for k, v in data.items()}  # {"hola": 1284, "mundo": 339}
```

**Conceptos clave:**

#### Obtener la primera clave de un diccionario
Los diccionarios no se acceden con `[0]`. Para obtener la primera clave:
```python
primera_clave = next(iter(diccionario.keys()))
```

#### Crear un diccionario invertido
Para invertir un diccionario (intercambiando claves y valores):
```python
diccionario_invertido = {v: k for k, v in diccionario.items()}
```

#### try/except para detectar tipos
Usa try/except para intentar convertir un valor a otro tipo:
```python
try:
    int(valor)
    # Es un número
except (ValueError, TypeError):
    # No es un número
```

---

### 8.3. Implementar search_exact

**Qué hace:**
Busca un token exacto en el diccionario y devuelve su ID.

**Ejemplo:**

Si tienes:
```python
self.token_to_id = {"hola": 1284, "mundo": 339}
```

Y buscas `"hola"`, debería devolver `[1284]`.

**¿Cómo funciona paso a paso?**

1. **Recibe un token** — Por ejemplo, `"hola"`
2. **Busca el token en el diccionario** — Usa `.get(token)`
3. **Si existe, devuelve el ID como lista** — `[token_id]`
4. **Si no existe, devuelve una lista vacía** — `[]`

**Conceptos clave:**

#### Buscar en un diccionario
Usa `.get()` para buscar un valor en un diccionario:
```python
valor = diccionario.get(clave)
# Si la clave no existe, devuelve None
```

#### Devolver listas
Aunque solo haya un elemento, devuelve una lista para consistencia:
```python
return [token_id]  # Si existe
return []  # Si no existe
```

---

### 8.4. Implementar search_characters

**Qué hace:**
Busca todos los tokens que están **compuestos solo por ciertos caracteres** y devuelve sus IDs.

**¿Por qué es importante?**

En la decodificación restringida, necesitas saber qué tokens son válidos en cada paso. Por ejemplo:
- Para extraer un número, necesitas saber qué tokens son dígitos (`0-9`)
- Para extraer un número con decimales, necesitas saber qué tokens son dígitos o el punto (`.`)

**Ejemplo concreto (no relacionado con el proyecto)**

Imagina que tienes un diccionario de tokens:

```
"0" → 10
"1" → 11
"2" → 12
"." → 13
"a" → 14
"b" → 15
```

Si buscas todos los tokens compuestos solo por dígitos (`"0123456789"`), el resultado debería ser:
```
[10, 11, 12]  # "0", "1", "2"
```

**¿Cómo funciona paso a paso?**

1. **Recibe un string de caracteres permitidos** — Por ejemplo, `"0123456789"`
2. **Itera sobre todos los tokens** del diccionario
3. **Verifica si todos los caracteres del token están en los caracteres permitidos**
4. **Si coincide, guarda el ID** — Añade el ID a una lista
5. **Devuelve la lista de IDs** — Todos los IDs de tokens compuestos solo por esos caracteres

**Conceptos clave:**

#### Verificar si todos los caracteres están permitidos
Usa `all()` con un generador:
```python
all(c in caracteres_permitidos for c in token)
```

**Ejemplo:**
```python
token = "0"
caracteres_permitidos = "0123456789"
all(c in caracteres_permitidos for c in token)  # True

token = "a"
caracteres_permitidos = "0123456789"
all(c in caracteres_permitidos for c in token)  # False
```

#### Error común: `valid_chars in token`
**Incorrecto:**
```python
if valid_chars in token:  # Verifica si el string completo está en el token
```

**Correcto:**
```python
if all(c in valid_chars for c in token):  # Verifica si todos los caracteres del token están en valid_chars
```

---

### 8.5. Implementar search_any_character

**Qué hace:**
Busca todos los tokens que contienen **al menos uno** de los caracteres dados.

**¿Por qué es importante?**

En el proyecto se usa para localizar los **terminadores de string**: tokens que contienen una comilla (`"`) o un salto de línea (`\n`). Cuando el modelo elige uno de ellos, sabemos que el string ha terminado.

**Diferencia con search_characters:**

| Función | Condición | Con `valid_chars = '"'` |
|---|---|---|
| `search_characters` | **Todos** los caracteres del token están permitidos | solo tokens como `"` |
| `search_any_character` | **Al menos uno** de los caracteres está en el token | `"`, `".`, `abc"`, `"` + salto... |

Es decir: `search_characters` sirve para "tokens hechos *solo* de esto"; `search_any_character`, para "tokens que *tocan* esto en algún sitio".

**¿Cómo funciona paso a paso?**

1. **Recibe un string de caracteres a buscar** — Por ejemplo, `'"\n'`
2. **Itera sobre todos los tokens** del diccionario
3. **Si el token contiene alguno de esos caracteres, guarda su ID**
4. **Devuelve la lista de IDs**

**Conceptos clave:**

#### all vs any
La única diferencia con `search_characters` es la función que comprueba la condición:
```python
all(c in valid_chars for c in token)   # search_characters: TODOS
any(c in valid_chars for c in token)   # search_any_character: ALGUNO
```

---

## LECCIÓN 9: Crear el decodificador restringido (constrained_decoder.py)

### 9.1. Implementar get_next_token_logits

**Qué hace:**
Pide al modelo las puntuaciones (logits) para el siguiente token.

**¿Por qué es importante?**

El modelo devuelve puntuaciones para cada token del vocabulario. Necesitas estas puntuaciones para saber qué token es el más probable.

**¿Qué parámetros recibe?**

1. **`model`** — El modelo (objeto Small_LLM_Model)
2. **`input_ids`** — Una lista de IDs de tokens (el contexto actual)

**¿Qué debe hacer paso a paso?**

1. **Llamar al SDK** — Usa `model.get_logits_from_input_ids(input_ids)` para obtener los logits
2. **Convertir a NumPy** — Usa `np.array(logits)` para convertir la lista a un array de NumPy
3. **Devolver el array** — Retorna el array de NumPy

**¿De dónde salen los input_ids?**

Los `input_ids` vienen de **convertir texto a IDs** usando el método `encode` del SDK:

```python
prompt = "What is the sum of 2 and 3?"
input_ids = model.encode(prompt)
# Resultado: [[892, 318, 262, 4771, 286, 16, 290, 17, 30]]
```

**Flujo completo:**

```
prompt_builder.py → texto → model.encode() → input_ids → get_next_token_logits() → logits
```

**Conceptos clave:**

#### encode vs get_logits_from_input_ids
- `encode(text)` — Convierte texto a IDs
- `get_logits_from_input_ids(input_ids)` — Devuelve puntuaciones para el siguiente token

#### Convertir a NumPy
El SDK devuelve una lista de floats. Para trabajar con ella fácilmente, conviértela a un array de NumPy:
```python
logits_array = np.array(logits)
```

---

### 9.2. Implementar apply_mask

**Qué hace:**
Pone **-infinito** a los tokens que no son válidos. Así el modelo no puede elegirlos nunca.

**¿Por qué es importante?**

El modelo elige el token con la puntuación más alta. Si pones -infinito a los tokens inválidos, el modelo no puede elegirlos nunca.

**Ejemplo concreto (no relacionado con el proyecto)**

Imagina que tienes puntuaciones para 5 tokens:

```
Token "0" → puntuación 5.2
Token "1" → puntuación 3.1
Token "a" → puntuación 0.5
Token "b" → puntuación 0.1
Token "2" → puntuación 4.0
```

Si solo quieres permitir dígitos (`0`, `1`, `2`), aplicas la máscara:

```
Token "0" → puntuación 5.2 (válido, se queda)
Token "1" → puntuación 3.1 (válido, se queda)
Token "a" → -infinito (inválido, bloqueado)
Token "b" → -infinito (inválido, bloqueado)
Token "2" → puntuación 4.0 (válido, se queda)
```

Ahora el modelo solo puede elegir entre `0`, `1`, `2`.

**¿Qué parámetros recibe?**

1. **`logits`** — Array de NumPy con las puntuaciones
2. **`valid_ids`** — Lista de IDs válidos

**¿Qué debe hacer paso a paso?**

1. **Crear un array nuevo lleno de -infinito** — Todos los tokens empiezan bloqueados
2. **Para cada ID válido, copiar su puntuación original** — Los tokens válidos se desbloquean
3. **Devolver el array enmascarado**

**Conceptos clave:**

#### Crear un array lleno de -infinito
Usa `np.full()` para crear un array lleno de un valor:
```python
masked = np.full(len(logits), -np.inf)
```

#### Copiar puntuaciones de tokens válidos
Itera sobre los IDs válidos y copia sus puntuaciones:
```python
for valid_id in valid_ids:
    masked[valid_id] = logits[valid_id]
```

#### Error común: usar el índice incorrecto
**Incorrecto:**
```python
masked[valid_id] = logits[valid_ids]  # ❌ valid_ids es una lista
```

**Correcto:**
```python
masked[valid_id] = logits[valid_id]  # ✅ valid_id es un número
```

---

### 9.3. Implementar select_best_token

**Qué hace:**
Elige el token con la **puntuación más alta** del array enmascarado.

**¿Por qué es importante?**

Después de aplicar la máscara, los tokens inválidos tienen -infinito. El modelo debe elegir el token con la puntuación más alta entre los válidos.

**¿Qué parámetros recibe?**

1. **`masked_logits`** — Array de NumPy con las puntuaciones enmascaradas

**¿Qué debe hacer paso a paso?**

1. **Encontrar el token con la puntuación más alta** — Usa `np.argmax()`
2. **Devolver su ID** — El ID del token con la puntuación más alta

**Conceptos clave:**

#### np.argmax()
`np.argmax()` devuelve el **índice** del valor más alto en un array:
```python
masked_logits = np.array([5.2, 3.1, -np.inf, -np.inf, 4.0])
best_token_id = np.argmax(masked_logits)
# Resultado: 0 (el índice del valor 5.2)
```

---

### 9.4. Crear la clase JSONGenerator

**Qué es JSONGenerator?**

Es la clase que coordina todo el proceso de decodificación restringida. Usa las tres funciones que ya has creado (`get_next_token_logits`, `apply_mask`, `select_best_token`) para generar texto token a token, asegurándose de que solo se eligen tokens válidos.

**¿Qué métodos necesita?**

1. **`__init__`** — Guarda el modelo y el vocabulario y precalcula las máscaras de tokens válidos por tipo.
2. **`select_function_name`** — Elige la función más probable (9.5).
3. **`generate_json`** — Punto de entrada: devuelve el JSON completo como texto (9.6).
4. **`select_state`** — Lógica de la máquina de estados (9.6).

**¿Qué parámetros recibe el constructor?**

1. **`model`** — El modelo (objeto `Small_LLM_Model`)
2. **`vocab`** — El vocabulario (objeto `VocabIndex`)

**¿Qué hace el constructor?**

Además de guardar el modelo y el vocabulario, **precalcula los IDs de los tokens válidos** de cada tipo. Conviene hacerlo una sola vez aquí: recorrer los ~150.000 tokens del vocabulario es caro y no tiene sentido repetirlo en cada generación.

```python
def __init__(self, model: Small_LLM_Model, vocab: VocabIndex):
    self.model = model
    self.vocab = vocab

    # Números: dígitos, punto y signo menos, más sus terminadores
    digits = self.vocab.search_characters("0123456789")
    dot = self.vocab.search_characters(".")
    minus = self.vocab.search_characters("-")
    self.number_valid_ids = digits + dot + minus
    self.number_terminators = self.vocab.search_characters(" ,\n\t")
    self.number_valid_ids += self.number_terminators

    # Strings: tokens sin caracteres estructurales, más sus terminadores
    self.string_valid_ids = []
    for token, token_id in self.vocab.token_to_id.items():
        if not any(char in token for char in "{}[]\\"):
            self.string_valid_ids.append(token_id)
    self.string_terminators = self.vocab.search_any_character('"\n')
    self.string_valid_ids += self.string_terminators

    # Booleanos: solo los tokens exactos "true" y "false"
    self.true_id = self.vocab.search_exact("true")
    self.false_id = self.vocab.search_exact("false")
    self.bool_valid_ids = self.true_id + self.false_id
```

**¿Por qué se añaden los terminadores a los tokens válidos?**

Porque el modelo tiene que poder **elegir** el terminador. Si no estuviera entre los tokens válidos, `apply_mask` le pondría `-inf` y el valor no terminaría nunca (bucle infinito).

**¿Dónde se crea el vocabulario?**

Fuera de la clase, y se pasa como parámetro:

```python
model = Small_LLM_Model()
vocab = VocabIndex(model.get_path_to_vocab_file())
generator = JSONGenerator(model, vocab)
```

Así la clase no depende de cómo se construye el vocabulario: es más fácil de testear y de reutilizar.

**Conceptos clave:**

#### Los métodos usan `self`
Todos los métodos reciben `self` como primer parámetro, así que dentro tienen `self.model` y `self.vocab`. No hace falta volver a pasar el modelo: sería redundante.

```python
def select_function_name(self, prompt_ids: list, function_list: list):
    # Aquí dentro ya tenemos self.model y self.vocab
    pass
```

#### No ejecutar código suelto dentro de la clase
Las llamadas van dentro de un método. Una línea como `logs = get_next_token_logits(...)` "suelta" en el cuerpo de la clase (fuera de todo método) es un error: se ejecutaría al definir la clase y no tiene acceso a `self`.

---

### 9.5. Implementar select_function_name

**Qué hace:**
Elige la función más probable. Puntúa cada nombre candidato token a token y se queda con el de mayor puntuación.

**¿Qué parámetros recibe?**

1. **`prompt_ids`** — IDs del prompt de selección (el contexto).
2. **`function_list`** — Lista de objetos `FunctionDefinition` disponibles.

**¿Cómo funciona paso a paso?**

Para cada función `fun`:

1. **Copiar el contexto** — `ids = list(prompt_ids)`. Se copia para que el nombre de una función no contamine la puntuación de la siguiente.
2. **Puntuar su nombre token a token** — Para cada token del nombre:
   - Calcular los logits del siguiente token en el contexto `ids`.
   - Convertirlos a log-probabilidades con `log_softmax`.
   - Sumar la log-probabilidad del token que forma parte del nombre a `score`.
   - Añadir ese token a `ids` (para que el siguiente token se evalúe con el contexto correcto).
3. **Comparar** — Si `score` supera el mejor hasta ahora, guardar esa función.

**¿Por qué log-probabilidades y no los logits directamente?**

Porque estamos sumando números de **posiciones distintas** y eso es engañoso:

- **Sesgo por longitud**: un nombre de 4 tokens suma 4 logits y otro de 2 suma solo 2. El primero puede "ganar" simplemente por tener más tokens, aunque cada uno sea improbable.
- **Escalas distintas**: un logit de +8 en una posición y uno de +7 en otra no son comparables tal cual.

Las **log-probabilidades** sí se pueden sumar con sentido: sumar logs equivale a multiplicar probabilidades (la probabilidad de toda la secuencia). `log_softmax` normaliza cada posición y elimina esa escala arbitraria.

**¿Por qué empezamos en `ids = list(prompt_ids)` y vamos añadiendo tokens?**

Porque el nombre de una función es una **secuencia**: el segundo token del nombre debe evaluarse *después* del primero. Si puntuáramos cada token siempre en el mismo contexto (`prompt_ids`), estaríamos tratando cada token como si fuese el primero del nombre, lo cual no refleja cómo generaría el modelo.

**Ejemplo concreto (no relacionado con el proyecto):**

Dos candidatas, `casa` (2 tokens) y `casita` (3 tokens). Ambas empiezan igual, pero los siguientes tokens se evalúan en contextos distintos. Sumando log-probabilidades gana la que el modelo considera globalmente más probable, sin que gane solo por tener más tokens.

**Conceptos clave:**

#### Copiar una lista
```python
ids = list(prompt_ids)
```
Crea una lista nueva. Con `ids = prompt_ids` modificaríamos el `prompt_ids` original y contaminaríamos la siguiente función.

#### Sumar log-probabilidades
```python
score += float(log_softmax(logits)[token_id])
```
`log_softmax(logits)[token_id]` es la log-probabilidad de que el modelo eligiera `token_id` en ese paso.

#### Recorrer los tokens del nombre
```python
for token_id in self.model.encode(fun.name).flatten().tolist():
    ...
```

---

### 9.6. Implementar generate_json y la máquina de estados

`generate_json` es el punto de entrada del generador: recibe el prompt y la lista de funciones y devuelve el JSON completo como texto. Por dentro usa una **máquina de estados** (FSM, *Finite State Machine*).

**Paso a paso:**

1. **Construir el prompt de selección** — `fun_selection(prompt, fun_list)` y convertirlo a IDs con `model.encode(...).flatten().tolist()`.
2. **Elegir la función** — `select_function_name(prompt_ids, fun_list)` (ver 9.5). Ya sabemos `selected_fun`.
3. **Forzar la apertura del JSON** — Se escriben directamente los tokens de `{"prompt": ..., "fn_name": "`.
4. **Bucle de estados** — Mientras el estado no sea `FINISHED`, `select_state` devuelve los tokens del estado actual; se añaden al contexto y se pasa al siguiente estado.
5. **Decodificar** — `model.decode(output_tokens)` convierte los IDs a texto. Ese texto es el JSON final.

**¿Por qué forzamos el prompt? (criterio importante)**

El campo `prompt` del JSON es exactamente la pregunta del usuario, que **ya conocemos**. Si dejamos que el modelo lo reescriba, tiende a cambiar mayúsculas y a reformularlo (por ejemplo, convertía `Greet shrek` en `Greet Shrek`). Ese eco acababa contaminando la extracción de otros argumentos (el `name` salía como `Shrek` en vez de `shrek`).

Por eso escribimos la apertura directamente:

```python
opening = '{"prompt": ' + json.dumps(prompt) + ', "fn_name": "'
```

`json.dumps` se encarga de escapar las comillas y los caracteres especiales del prompt (importante cuando el prompt ya contiene comillas, como en `fn_substitute_string_with_regex`). Esto **no es hacer trampa**: es la misma idea de la decodificación restringida — si un token es conocido, se fuerza; al modelo solo se le pide lo que no sabemos (el valor de los argumentos).

**Los estados de la FSM:**

| Estado | ¿Forzado o modelo? | Qué emite |
|---|---|---|
| `FN_NAME` | Forzado | El nombre de la función elegida |
| `AFTER_NAME` | Forzado | `", "args": {`, o `", "args": {}` si no hay parámetros |
| `PARAM_KEY` | Forzado | La clave del parámetro (`"param": `, o `"param": "` si es string) |
| `PARAM_VALUE` | **Modelo** | El valor, respetando la máscara de su tipo |
| `NEXT_PARAM` | Forzado | `, "siguiente": `, o el cierre (`}` / `"}`) si ya no quedan parámetros |
| `END` | Forzado | `}` (cierra el objeto exterior) |
| `FINISHED` | — | Fin |

Fíjate en la columna central: **el modelo solo decide en `PARAM_VALUE`**. Todo lo demás (estructura, nombres de claves y nombre de la función) lo escribimos nosotros.

**Cómo se decide el valor en `PARAM_VALUE`:**

1. Se elige la máscara según el tipo del parámetro (`number`, `string` o `boolean`).
2. `get_next_token_logits` → `apply_mask` → `select_best_token`.
3. Si el token elegido es un **terminador**, el valor ha acabado: se devuelve `[]` y se pasa a `NEXT_PARAM`.
4. Si no, se devuelve `[token]` y se sigue en `PARAM_VALUE`.

**Un detalle importante: los terminadores no son iguales en todos los tipos**

- **`number`**: los terminadores son tokens compuestos **solo** por ` ,\n\t` (espacio, coma, tabulador, salto). Así el número "acaba" justo al llegar a la coma que separa el siguiente parámetro.
- **`string`**: se usan tokens que **contienen** `"` o `\n` (búsqueda con `any`, ver 8.5). Cualquier token donde aparezca una comilla cierra el string.
- **`boolean`**: no hay terminador; el valor es exactamente un token (`true` o `false`) y se pasa directamente al siguiente estado.

Esta asimetría (`search_characters` para números, `search_any_character` para strings) es intencionada: un número no debe "tragarse" la coma, pero un string puede terminar con una comilla pegada a texto.

**¿Cómo se implementa `select_state`?**

Es un `match` sobre el estado que **siempre** devuelve una tupla `(tokens, siguiente_estado)`:

```python
match state:
    case JSONState.FN_NAME:
        tokens = self.model.encode(selected_fun.name).flatten().tolist()
        return tokens, JSONState.AFTER_NAME
    # ... una rama por estado ...
```

Cada rama construye el texto forzado, lo convierte a IDs con `model.encode` y lo devuelve junto al siguiente estado. La única rama que consulta al modelo es `PARAM_VALUE`.

> La rama `FINISHED` nunca se ejecuta: el bucle `while state != JSONState.FINISHED` corta antes. Es solo una salvaguarda.

---

## LECCIÓN 10: Crear el orquestador (function_caller.py)

### 10.1. ¿Qué es FunctionCaller?

Es el **orquestador** que coordina todo el proceso. Es como el director de orquesta: no toca ningún instrumento, pero sabe cuándo debe entrar cada uno.

**¿Por qué es importante?**

Sin FunctionCaller, los otros módulos (prompt_builder, constrained_decoder, tools) no pueden trabajar juntos. FunctionCaller es el que los une y hace que funcionen como un equipo.

**¿Qué hace el constructor?**

1. **Recibe el modelo** — Lo guarda como `self.model`
2. **Crea el vocabulario** — Usa `VocabIndex` con la ruta del modelo
3. **Crea el generador** — Usa `JSONGenerator` con el modelo y el vocabulario

```python
def __init__(self, model: Small_LLM_Model):
    self.model = model
    vocab_path = model.get_path_to_vocab_file()
    self.vocab = VocabIndex(vocab_path)
    self.generator = JSONGenerator(model, self.vocab)
```

---

### 10.2. Implementar resolve

**Qué hace:**
Dado un prompt y la lista de funciones, genera el JSON con el decoder y devuelve un `FunctionCall`.

**¿Qué parámetros recibe?**

1. **`prompt`** — La pregunta del usuario (string)
2. **`fun_list`** — La lista de funciones disponibles (lista de `FunctionDefinition`)

**¿Qué debe hacer paso a paso?**

1. **Generar el JSON** — `raw_json = self.generator.generate_json(prompt, fun_list)`. Toda la lógica (selección de función y extracción de argumentos) vive dentro del generador.
2. **Parsear** — `data = json.loads(raw_json)`. Si el texto no es JSON válido, se lanza un `ValueError`.
3. **Ensamblar el resultado** — `FunctionCall(prompt=prompt, fn_name=data["fn_name"], args=data["args"])`.

**Flujo de datos completo:**

```
prompt (str) + fun_list (list[FunctionDefinition])
    ↓
generator.generate_json(prompt, fun_list) → raw_json (str)
    ↓
json.loads(raw_json) → data (dict)
    ↓
FunctionCall(prompt=prompt, fn_name=data["fn_name"], args=data["args"])
```

**De dónde viene cada dato:**

| Dato | De dónde viene |
|------|----------------|
| `prompt` | La pregunta del usuario (string) |
| `fun_list` | El archivo `functions_definition.json` (lista de `FunctionDefinition`) |
| `raw_json` | `generate_json()` (string con el JSON) |
| `data` | `json.loads()` (diccionario de Python) |
| `FunctionCall` | El resultado final con `prompt`, `fn_name` y `args` |

**Conceptos clave:**

#### Qué aporta cada módulo
- **prompt_builder** — Construye el prompt de selección.
- **constrained_decoder** — Genera el JSON completo (función + argumentos).
- **tools** — Lee los archivos de entrada y escribe los resultados.

#### El resultado tiene exactamente tres claves
`prompt`, `fn_name` y `args`. El formato de salida es fijo:

```json
{"prompt": "What is the sum of 2 and 3?", "fn_name": "fn_add_numbers", "args": {"a": 2, "b": 3}}
```

#### Usa el prompt original
`FunctionCall.prompt` es la pregunta **original** del usuario. El prompt de selección (`fun_selection`) es solo un texto interno que se le pasa al modelo; no debe acabar en el resultado.

---

## LECCIÓN 11: Crear el punto de entrada (__main__.py)

### 11.1. Implementar parse_args

**Qué hacer:**
Definir con `argparse` los argumentos `--input` (directorio de entrada) y `--output` (archivo de salida), y devolverlos.

**Por qué lo haces:**
El programa necesita saber de dónde leer y dónde escribir.

---

### 11.2. Implementar main

**Qué hacer:**
Coordinar todo el flujo:
1. Parsear los argumentos
2. Cargar las definiciones de funciones y los prompts
3. Cargar el modelo
4. Crear el `FunctionCaller`
5. Llamar a `caller.resolve()` para cada prompt
6. Escribir los resultados y mostrar un resumen

---

### 11.3. Añadir el bloque if __name__ == "__main__"

**Qué hacer:**
Añadir el bloque estándar de Python:
```python
if __name__ == "__main__":
    sys.exit(main())
```

**Por qué lo haces:**
Hace que `main()` se ejecute solo cuando lanzas el módulo (`python -m src`), no cuando lo importas. El código de salida es 0 si todo fue bien y 1 si hubo errores.

---

## LECCIÓN 12: Ejecutar y depurar el programa

### 12.1. Ejecutar el programa

**Qué hacer:**
Ejecuta `make run` (o `uv run python -m src`). El programa cargará el modelo, procesará todas las preguntas y escribirá `data/output/function_calling_results.json`.

---

### 12.2. Comprobar los resultados

Abre `data/output/function_calling_results.json` y revisa cada resultado. Conviene comprobar, en este orden:

1. **¿Es JSON parseable?** — Debe poder leerse con `json.loads`. Si esto falla, el fallo es del decoder.
2. **¿Tiene las tres claves exactas?** — `prompt`, `fn_name` y `args` (ni una más ni una menos).
3. **¿El `prompt` es idéntico a la pregunta original?** — Sin cambios de mayúsculas ni reformulaciones.
4. **¿El `fn_name` es correcto?** — Debe ser una de las funciones disponibles y la adecuada para la pregunta (aquí puede fallar el modelo).
5. **¿Los `args` son correctos y del tipo adecuado?** — Números como números (no `"2"`), strings bien formados, etc.

Recuerda el reparto de responsabilidades de la Lección 3.5: si el JSON está bien formado pero el valor no es el esperado, el fallo es del modelo, no del decoder.

---

### 12.3. Depurar errores

**Qué hacer:**
Ejecuta el programa bajo `pdb` para ver dónde falla:
```bash
uv run python -m pdb -m src
```
(Si tu Makefile define el target, `make debug` equivale a lo mismo.)

Con `pdb` puedes avanzar paso a paso y ver los valores de las variables.

---

# GLOSARIO

| Término | Significado |
|---------|-------------|
| **Token** | Trozo de texto que el modelo entiende (palabra, parte de palabra, signo) |
| **ID** | Número único que identifica un token |
| **Logit** | Puntuación en bruto que el modelo asigna a cada token |
| **Softmax** | Fórmula que convierte logits en probabilidades (suma = 1) |
| **Log-softmax** | Logaritmo de la softmax, más estable numéricamente |
| **Argmax** | Operación que elige el elemento con valor máximo |
| **Máscara** | Array que bloquea ciertos tokens poniéndoles -infinito |
| **Terminador** | Token que marca el final de un valor (p. ej. una comilla en un string) |
| **Enum** | Tipo con un conjunto fijo de valores con nombre |
| **Prompt** | Texto que le pasamos al modelo para guiar su respuesta |
| **Inferencia** | Cuando el modelo genera texto (a diferencia de entrenamiento) |
| **Pydantic** | Librería que valida que los datos tengan el formato correcto |
| **NumPy** | Librería para trabajar con arrays de números de forma eficiente |
| **FSM** | Finite State Machine (Autómata Finito) — máquina que rastrea en qué estado estamos |
| **Decodificación restringida** | Técnica de filtrar tokens inválidos durante la generación |
| **EOS token** | End-Of-Sequence — token especial que indica que el texto ha terminado |
| **Tensor** | Array multidimensional (como una tabla de números) usado por PyTorch |

---

**Fin de la guía. ¡Mucha suerte con el proyecto!**
