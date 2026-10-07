# GUÍA UNIVERSITARIA: Call Me Maybe

## Una clase paso a paso para entender y construir el proyecto

**Duración estimada:** 4-5 horas

---

# LECCIÓN 1: EL PROBLEMA QUE VAMOS A RESOLVER (30 minutos)

## 1.1. El mundo antes de este proyecto

Imagina que trabajas en una empresa y te piden crear un sistema que entienda lenguaje humano. Por ejemplo, un usuario escribe:

> "¿Cuánto es 40 más 2?"

Y tu sistema debe ser capaz de:
1. Entender que quiere sumar
2. Identificar los números 40 y 2
3. Llamar a una función de suma con esos valores

Eso suena fácil, pero hay un problema fundamental: los ordenadores no entienden el lenguaje humano. Solo entienden instrucciones muy precisas y estructuradas.

## 1.2. La solución obvia (y por qué no funciona)

La primera idea que se te podría ocurrir es usar un modelo de lenguaje como ChatGPT. Le preguntas "¿qué función debo llamar?" y te responde.

Pero aquí viene el problema: los modelos de lenguaje son **torpes con el formato**. Si le pides a un modelo que escriba JSON (un formato de datos muy estructurado), fallará muchas veces.

¿Por qué? Porque los modelos de lenguaje están diseñados para generar texto natural, no para seguir reglas estrictas de formato. Es como pedirle a un poeta que escriba un formulario de Hacienda: puede que lo haga bien, pero probablemente se invente cosas o se salte campos.

## 1.3. El problema específico de los modelos pequeños

En este proyecto no vamos a usar un modelo enorme como GPT-4. Vamos a usar un modelo pequeño llamado Qwen3-0.6B, que tiene solo 600 millones de parámetros.

Para que te hagas una idea:
- GPT-4 tiene más de 1 billón de parámetros
- Nuestro modelo tiene 600 millones
- Es como comparar un autobús con una bicicleta

Los modelos pequeños son mucho más torpes con el formato. Si le pides a Qwen3-0.6B que genere JSON directamente, fallará el 70% de las veces. Generará cosas como:

```
{"function": "sumar"}   ← le falta el campo "argumentos"
```

O peor:

```
La función es sumar con a=40 y b=2   ← esto no es JSON
```

## 1.4. La solución: no dejar que se equivoque

La idea clave del proyecto es: **no le preguntes al modelo qué quiere escribir. Oblígalo a escribir solo lo que es válido.**

En lugar de pedirle al modelo "escribe un JSON con la función y los argumentos", vamos a hacer algo mucho más inteligente:

1. Le preguntamos al modelo qué token (palabra o parte de palabra) quiere escribir a continuación
2. Antes de que elija, le decimos: "solo puedes elegir entre estos 3 tokens que son válidos"
3. El modelo elige el que más le guste de esos 3
4. Repetimos el proceso para el siguiente token

Así es **físicamente imposible** que el modelo escriba algo incorrecto. Es como si a un niño le das a elegir entre "manzana", "pera" y "plátano" — puede elegir cualquiera, pero nunca va a elegir "coche" porque no está entre las opciones.

Esta técnica se llama **decodificación restringida** (constrained decoding en inglés) y es el corazón del proyecto.

---

# LECCIÓN 2: CÓMO FUNCIONA UN MODELO DE LENGUAJE POR DENTRO (30 minutos)

## 2.1. Los tokens: los ladrillos del lenguaje

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

## 2.2. Input IDs: el modelo solo entiende números

Cuando le pasas texto al modelo, primero lo conviertes a números (IDs de tokens). Este proceso se llama **tokenización**.

Por ejemplo, si le pasas "Hola mundo", el modelo no ve las letras H-o-l-a-m-u-n-d-o. Ve una lista de números: [1284, 339].

Esa lista de números es lo que el modelo recibe. Se llama **input_ids**.

## 2.3. Logits: las puntuaciones del modelo

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

## 2.4. Softmax: convertir puntuaciones en probabilidades

Los logits son números en bruto, difíciles de interpretar. Para convertirlos en probabilidades (números entre 0 y 1 que suman 1), se usa **softmax**.

La fórmula es: softmax(x_i) = e^(x_i) / suma(e^(x_j))

No te asustes con la fórmula. Lo que hace es:
1. Coge cada logit
2. Calcula el número e^logit (e ≈ 2.71828 elevado al logit)
3. Divide cada uno por la suma de todos

El resultado: probabilidades que suman 1. Un token con probabilidad 0.8 significa que el modelo cree que hay un 80% de probabilidades de que sea ese.

En el proyecto usamos **log-softmax**, que es el logaritmo de la softmax. Es más estable numéricamente y más práctico para sumar probabilidades (en vez de multiplicarlas).

## 2.5. Argmax: elegir el mejor

El argmax es una operación que elige el elemento con valor máximo. Si tienes [0.3, 2.1, 5.4, 0.1], el argmax devuelve el índice 2 (el token con puntuación 5.4).

---

# LECCIÓN 3: LA DECODIFICACIÓN RESTRINGIDA (30 minutos)

## 3.1. ¿Qué es?

La decodificación restringida es una técnica que filtra las opciones del modelo en cada paso para que solo pueda elegir tokens válidos.

## 3.2. Analogía: el menú del restaurante

Imagina que vas a un restaurante con un menú de 150 platos. El camarero (el modelo) te pregunta qué quieres.

**Sin restricción:** El camarero puede recomendarte cualquier plato, incluso los que no están en el menú.

**Con restricción:** Tú le dices "solo puedes elegir entre paella, sushi o pizza". Ahora es imposible que te recomiende una hamburguesa.

En el proyecto:
- El "menú" son los 150.000 tokens del vocabulario del modelo
- La "restricción" es: "solo puedes elegir tokens que mantengan el JSON válido"

## 3.3. ¿Cómo funciona paso a paso?

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

## 3.4. ¿Por qué -infinito y no 0?

Porque el modelo elige el token con el logit MÁS ALTO. Si pusiéramos 0, algunos tokens prohibidos podrían tener logits negativos (por ejemplo, -5) y el 0 sería mayor que esos, con lo que el token prohibido sería elegido.

Con -infinito, ningún token prohibido puede ser elegido jamás. Es la puntuación más baja posible.

---

# LECCIÓN 4: ARQUITECTURA DEL PROYECTO (30 minutos)

## 4.1. El mapa de los archivos

El proyecto tiene una estructura de carpetas que organiza el código de forma lógica:

- **src/**: Todo el código fuente está aquí
  - **__init__.py**: Marca la carpeta como un módulo Python
  - **__main__.py**: Punto de entrada (el que se ejecuta con "python -m src")
  - **models.py**: Las "plantillas" de datos (Pydantic)
  - **vocab.py**: El diccionario token ↔ ID
  - **constrained_decoder.py**: El CORAZÓN del proyecto (decodificación restringida)
  - **function_caller.py**: El ORQUESTADOR (coordina todo)
  - **prompt_builder.py**: Construye los mensajes para la IA
  - **tools.py**: Lee y escribe archivos

- **llm_sdk/**: El SDK (caja de herramientas) para usar el modelo
  - **llm_sdk/__init__.py**: Contiene la clase Small_LLM_Model

- **data/**: Los archivos de datos
  - **input/**: Los archivos de entrada (funciones y preguntas)
  - **output/**: Los archivos de salida (resultados)

- **Makefile**: Atajos para comandos frecuentes
- **pyproject.toml**: Configuración del proyecto Python

## 4.2. El flujo de principio a fin

Vamos a seguir el viaje de una pregunta desde que entra hasta que sale:

1. **LEER ARCHIVOS DE ENTRADA**: El programa lee las funciones disponibles y las preguntas de prueba
2. **CARGAR EL MODELO**: Se carga el modelo Qwen3-0.6B
3. **CONSTRUIR PROMPT DE SELECCIÓN**: Se construye un mensaje para el modelo que dice "estas son las funciones disponibles, ¿cuál quieres usar?"
4. **ELEGIR FUNCIÓN**: El modelo elige la función más apropiada usando decodificación restringida
5. **PARA CADA PARÁMETRO, EXTRAER VALOR**: Para cada parámetro de la función elegida, se construye un prompt y se extrae el valor usando decodificación restringida
6. **ENSAMBLAR RESULTADO**: Se junta todo en un objeto FunctionCall
7. **GUARDAR EN ARCHIVO**: Se escribe el resultado en un archivo JSON

---

# LECCIÓN 5: MANOS A LA OBRA — FASE 0 (30 minutos)

## Paso 5.1: Crear la estructura de carpetas

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

## Paso 5.2: Crear pyproject.toml

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

## Paso 5.3: Crear el Makefile

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
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

**Por qué lo haces:**
El Makefile te permite escribir `make run` en vez de `uv run python -m src --input data/input --output data/output/results.json`. Es más rápido y menos propenso a errores.

**Cómo debería funcionar:**
Cuando escribas `make run`, el programa debería ejecutarse. Si hay errores, el Makefile te mostrará mensajes de error claros.

---

## Paso 5.4: Copiar llm_sdk

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

## Paso 5.5: Crear los archivos de datos de entrada

**Qué hacer:**
Crea los archivos JSON con las funciones disponibles y las preguntas de prueba.

**Cómo hacerlo:**
Crea `data/input/function_definitions.json` con las funciones que el sistema puede usar. Cada función tiene un nombre, una descripción, parámetros y un tipo de retorno.

Crea `data/input/function_calling_tests.json` con las preguntas que el sistema debe procesar.

**Por qué lo haces:**
Estos archivos son la entrada del programa. Sin ellos, el programa no tiene nada que procesar.

**Cómo debería funcionar:**
Cuando el programa se ejecute, leerá estos archivos y procesará cada pregunta.

---

# LECCIÓN 6: MANOS A LA OBRA — FASE 1 (45 minutos)

## Paso 6.1: Entender qué son los modelos de datos y por qué existen

**Qué es:**
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
Piensa en BaseModel como un formulario de impuestos. El formulario tiene campos con reglas: "este campo debe ser un número", "este campo debe ser un texto". Si intentas escribir "hola" en el campo de número, el formulario te dice "error: este campo debe ser un número". BaseModel hace lo mismo pero en código.

---

## Paso 6.2: Crear ParameterDefinition

**Qué hacer:**
Define la plantilla para un parámetro de función.

**Qué necesita:**
- Un campo para el tipo del parámetro (por ejemplo, "number", "string", "boolean")

**Qué debe hacer:**
- Validar que el tipo sea un texto válido
- Lanzar un error si el tipo no es un texto

**Por qué lo haces:**
Cada parámetro de una función tiene un tipo. Esta plantilla asegura que el tipo sea un texto válido.

**Cómo debería funcionar:**
Si intentas crear un parámetro con un tipo que no es un texto, Pydantic lanzará un error claro.

---

## Paso 6.3: Crear FunctionDefinition

**Qué hacer:**
Define la plantilla para una función completa.

**Qué necesita:**
- Un campo para el nombre de la función
- Un campo para la descripción de la función
- Un campo para los parámetros de la función (un diccionario donde las claves son nombres de parámetros y los valores son objetos ParameterDefinition)
- Un campo para el tipo de retorno de la función

**Qué debe hacer:**
- Validar que todos los campos estén presentes y sean del tipo correcto
- Convertir los diccionarios JSON en objetos ParameterDefinition automáticamente
- Lanzar un error si algún campo es inválido

**Por qué lo haces:**
Una función tiene un nombre, una descripción, parámetros y un tipo de retorno. Esta plantilla asegura que todos estos campos estén presentes y sean del tipo correcto.

**Cómo debería funcionar:**
Si intentas crear una función sin nombre o con parámetros inválidos, Pydantic lanzará un error.

---

## Paso 6.4: Crear FunctionCall

**Qué hacer:**
Define la plantilla para el resultado final.

**Qué necesita:**
- Un campo para la pregunta original
- Un campo para el nombre de la función elegida
- Un campo para los argumentos extraídos (un diccionario donde las claves son nombres de parámetros y los valores son los argumentos)

**Qué debe hacer:**
- Validar que todos los campos estén presentes y sean del tipo correcto
- Lanzar un error si algún campo es inválido

**Por qué lo haces:**
El resultado final del programa es un objeto FunctionCall que contiene la pregunta original, el nombre de la función elegida y los argumentos extraídos.

**Cómo debería funcionar:**
Cuando el programa genera un resultado, crea un objeto FunctionCall con los datos correctos.

---

# LECCIÓN 7: MANOS A LA OBRA — FASE 2 (45 minutos)

## Paso 7.1: Entender qué es el vocabulario

**Qué es:**
El vocabulario es un diccionario que asocia tokens (texto) con IDs (números). El modelo solo entiende números, así que necesitas esta traducción.

**Por qué lo necesitas:**
Cuando el modelo devuelve logits, devuelve puntuaciones para cada ID del vocabulario. Para saber qué token corresponde a cada ID, necesitas el vocabulario.

---

## Paso 7.2: Cargar el archivo vocab.json

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
Cuando creas un objeto VocabIndex, carga el archivo y construye los diccionarios. Después, puedas buscar tokens e IDs rápidamente.

---

## Paso 7.3: Implementar métodos de búsqueda

**Qué hacer:**
Implementa métodos para buscar tokens exactos, tokens que empiezan por un prefijo y tokens compuestos por ciertos caracteres.

**Qué necesita:**
- Un texto o prefijo para buscar

**Qué debe hacer:**
- Buscar en el diccionario y devolver los IDs que cumplen la condición
- Si no hay resultados, devolver una lista vacía

**Por qué lo haces:**
En la decodificación restringida, necesitas saber qué tokens son válidos en cada paso. Por ejemplo, para extraer un número, necesitas saber qué tokens son dígitos.

**Cómo debería funcionar:**
Cuando llamas al método con un prefijo, devuelve todos los IDs de tokens que empiezan por ese prefijo.

---

# LECCIÓN 8: MANOS A LA OBRA — FASE 3 (60 minutos)

## Paso 8.1: Entender qué es la decodificación restringida

**Qué es:**
La decodificación restringida es una técnica que filtra las opciones del modelo en cada paso para que solo pueda elegir tokens válidos.

**Por qué la necesitas:**
El modelo Qwen3-0.6B es pequeño y falla el 70% de las veces cuando se le pide JSON. La decodificación restringida hace que sea físicamente imposible que el modelo escriba algo incorrecto.

---

## Paso 8.2: Implementar get_next_token_logits

**Qué hacer:**
Implementa una función que pide al modelo las puntuaciones para el siguiente token.

**Qué necesita:**
- El modelo
- Una lista de IDs de tokens

**Qué debe hacer:**
- Pasar los IDs al modelo
- Obtener los logits del modelo
- Convertir los logits a un array de NumPy
- Devolver el array de logits

**Por qué lo haces:**
El modelo devuelve puntuaciones para cada token del vocabulario. Necesitas estas puntuaciones para saber qué token es el más probable.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve una lista de 150.000 números (uno por cada token del vocabulario).

---

## Paso 8.3: Implementar apply_mask

**Qué hacer:**
Implementa una función que pone -infinito a los tokens que no son válidos.

**Qué necesita:**
- Los logits del modelo
- Una lista de IDs válidos

**Qué debe hacer:**
- Crear un array nuevo lleno de -infinito
- Para cada ID válido, copiar su logit original a esa posición
- Devolver el array enmascarado

**Por qué lo haces:**
El modelo elige el token con la puntuación más alta. Si pones -infinito a los tokens inválidos, el modelo no puede elegirlos nunca.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve una lista donde los tokens inválidos tienen -infinito y los válidos tienen su puntuación original.

---

## Paso 8.4: Implementar select_best_token

**Qué hacer:**
Implementa una función que elija el token con la puntuación más alta.

**Qué necesita:**
- Los logits enmascarados

**Qué debe hacer:**
- Encontrar el token con la puntuación más alta
- Devolver su ID
- Si todos son -infinito, devolver None

**Por qué lo haces:**
Después de aplicar la máscara, necesitas saber qué token es el más probable entre los válidos.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve el ID del token con la puntuación más alta. Si todos son -infinito, devuelve None.

---

## Paso 8.5: Implementar la clase JSONGenerator

**Qué hacer:**
Crea una clase que coordine todo el proceso de generación con restricciones.

**Qué necesita:**
- El modelo
- El vocabulario

**Qué debe hacer:**
- Tener un método para seleccionar el nombre de una función
- Tener un método para extraer un número
- Tener un método para extraer un string
- Tener un método para extraer un boolean
- En cada paso, aplicar la máscara para que el modelo solo pueda elegir tokens válidos

**Por qué lo haces:**
Esta clase es el corazón del proyecto. Contiene toda la lógica de la decodificación restringida.

**Cómo debería funcionar:**
Cuando llamas a un método, el modelo genera texto token a token, y en cada paso solo puede elegir tokens válidos.

---

# LECCIÓN 9: MANOS A LA OBRA — FASE 4 (45 minutos)

## Paso 9.1: Entender qué es un prompt

**Qué es:**
Un prompt es el texto que le pasamos al modelo para guiar su respuesta.

**Por qué lo necesitas:**
La forma en que le preguntamos al modelo afecta mucho a la respuesta. Un buen prompt hace que el modelo entienda exactamente qué quieres.

---

## Paso 9.2: Implementar build_function_selection_prompt

**Qué hacer:**
Construye un prompt que pregunte al modelo qué función quiere usar.

**Qué necesita:**
- La pregunta del usuario
- La lista de funciones disponibles

**Qué debe hacer:**
- Construir un texto que explique al modelo qué debe hacer
- Incluir la lista de funciones disponibles
- Terminar el prompt con "Function to call: " para que el modelo "quiera" continuar con el nombre de una función

**Por qué lo haces:**
El prompt termina con "Function to call: " para que el modelo "quiera" continuar con el nombre de una función.

**Cómo debería funcionar:**
Cuando el modelo recibe este prompt, genera el nombre de la función más apropiada.

---

## Paso 9.3: Implementar build_argument_extraction_prompt

**Qué hacer:**
Construye un prompt que pregunte al modelo qué valor tiene un parámetro.

**Qué necesita:**
- La pregunta del usuario
- La función elegida
- El parámetro que quieres extraer
- Los argumentos ya extraídos (para dar contexto al modelo)

**Qué debe hacer:**
- Construir un texto que explique al modelo qué debe hacer
- Incluir la información de la función y el parámetro
- Si ya hay argumentos extraídos, incluirlos como contexto
- Terminar el prompt con "Value: " para que el modelo "quiera" continuar con el valor del parámetro

**Por qué lo haces:**
El prompt termina con "Value: " para que el modelo "quiera" continuar con el valor del parámetro.

**Cómo debería funcionar:**
Cuando el modelo recibe este prompt, genera el valor del parámetro.

---

# LECCIÓN 10: MANOS A LA OBRA — FASE 5 (30 minutos)

## Paso 10.1: Entender qué es el orquestador

**Qué es:**
El orquestador es la clase que coordina todo el proceso. Es como el director de orquesta: no toca ningún instrumento, pero sabe cuándo debe entrar cada uno.

**Por qué lo necesitas:**
El proceso tiene varios pasos: elegir función, extraer argumentos, ensamblar resultado. El orquestador se encarga de que estos pasos se ejecuten en el orden correcto.

---

## Paso 10.2: Implementar el método resolve

**Qué hacer:**
Implementa un método que recibe una pregunta y una lista de funciones, y devuelve un objeto FunctionCall.

**Qué necesita:**
- La pregunta del usuario
- La lista de funciones disponibles
- El decodificador restringido
- El constructor de prompts

**Qué debe hacer:**
1. Construir el prompt de selección de función
2. Elegir la función usando el decodificador
3. Para cada parámetro de la función:
   - Construir el prompt de extracción
   - Extraer el valor usando el decodificador
4. Ensamblar el resultado en un objeto FunctionCall

**Por qué lo haces:**
Este método es el punto de entrada del proceso. Cuando lo llamas, el sistema completo se pone en marcha.

**Cómo debería funcionar:**
Cuando llamas al método, devuelve un objeto FunctionCall con la pregunta original, el nombre de la función elegida y los argumentos extraídos.

---

# LECCIÓN 11: MANOS A LA OBRA — FASE 6 (30 minutos)

## Paso 11.1: Implementar json_reader

**Qué hacer:**
Implementa una función que lea un archivo JSON y lo convierta en datos Python.

**Qué necesita:**
- La ruta al archivo JSON

**Qué debe hacer:**
- Verificar que el archivo exista
- Abrir el archivo
- Cargar el JSON
- Devolver los datos como objetos Python

**Por qué lo haces:**
Necesitas leer los archivos de entrada (funciones y preguntas) para que el programa pueda procesarlos. Sin esta función, el programa no puede leer nada.

**Para qué sirve:**
Es la base de todas las demás funciones de lectura. `load_function_def` y `load_prompt` la usan para leer sus archivos.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve los datos del archivo JSON como objetos Python.

---

## Paso 11.2: Implementar load_function_def

**Qué hacer:**
Implementa una función que cargue las definiciones de funciones y las valide con Pydantic.

**Qué necesita:**
- La ruta al archivo JSON con las definiciones de funciones

**Qué debe hacer:**
- Leer el JSON
- Verificar que sea una lista
- Para cada elemento, crear un objeto FunctionDefinition
- Si alguna definición es inválida, indicar en qué índice falló

**Por qué lo haces:**
Necesitas validar que las funciones tengan el formato correcto antes de usarlas. Si una función tiene un error (por ejemplo, le falta un campo), el programa debe detectarlo inmediatamente, no fallar más tarde de forma misteriosa.

**Para qué sirve:**
Convierte los datos JSON en objetos `FunctionDefinition` que el resto del programa puede usar. Sin esta función, el programa no puede usar las funciones disponibles.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve una lista de objetos FunctionDefinition.

---

## Paso 11.3: Implementar load_prompt

**Qué hacer:**
Implementa una función que cargue los prompts de prueba.

**Qué necesita:**
- La ruta al archivo JSON con los prompts

**Qué debe hacer:**
- Leer el JSON
- Verificar que sea una lista
- Para cada elemento, extraer el texto del prompt
- Devolver una lista de strings

**Por qué lo haces:**
Necesitas leer las preguntas de prueba para que el programa pueda procesarlas. Los prompts pueden venir en diferentes formatos (string o diccionario), así que la función debe manejar ambos casos.

**Para qué sirve:**
Convierte los datos JSON en una lista de strings que el resto del programa puede usar. Sin esta función, el programa no puede leer las preguntas de prueba.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve una lista de strings.

---

## Paso 11.4: Implementar json_exporter

**Qué hacer:**
Implementa una función que escriba los resultados en un archivo JSON.

**Qué necesita:**
- La lista de objetos FunctionCall
- La ruta al archivo de salida

**Qué debe hacer:**
- Crear la carpeta si no existe
- Convertir los objetos FunctionCall a diccionarios
- Escribir el JSON en el archivo

**Por qué lo haces:**
Necesitas guardar los resultados para que el usuario pueda verlos. Los objetos Pydantic no se pueden escribir directamente a JSON, hay que convertirlos a diccionarios primero.

**Para qué sirve:**
Guarda los resultados del programa en un archivo JSON. Sin esta función, el programa no puede guardar los resultados.

**Cómo debería funcionar:**
Cuando llamas a la función, crea el archivo JSON con los resultados.

---

# LECCIÓN 12: MANOS A LA OBRA — FASE 7 (30 minutos)

## Paso 12.1: Implementar parse_args

**Qué hacer:**
Implementa una función que defina los argumentos que acepta el programa.

**Qué necesita:**
- Ninguno (usa argparse de la librería estándar)

**Qué debe hacer:**
- Definir el argumento --input (directorio de entrada)
- Definir el argumento --output (archivo de salida)
- Devolver los argumentos parseados

**Por qué lo haces:**
El programa necesita saber dónde están los archivos de entrada y dónde guardar los resultados.

**Cómo debería funcionar:**
Cuando ejecutas el programa con --input y --output, el programa usa esas rutas.

---

## Paso 12.2: Implementar main

**Qué hacer:**
Implementa la función principal que coordina todo el proceso.

**Qué necesita:**
- Los argumentos parseados
- Las rutas a los archivos de entrada y salida

**Qué debe hacer:**
1. Parsear los argumentos
2. Cargar los archivos de entrada
3. Cargar el modelo
4. Crear el FunctionCaller
5. Para cada prompt, llamar a caller.resolve()
6. Escribir los resultados
7. Mostrar un resumen

**Por qué lo haces:**
Esta función es el punto de entrada del programa. Cuando la llamas, todo el sistema se pone en marcha.

**Cómo debería funcionar:**
Cuando ejecutas el programa, procesa todas las preguntas y genera un archivo JSON con los resultados.

---

## Paso 12.3: Añadir el bloque if __name__ == "__main__"

**Qué hacer:**
Añade el bloque estándar de Python que ejecuta main() cuando el archivo se ejecuta directamente.

**Qué necesita:**
- Ninguno

**Qué debe hacer:**
- Verificar si el archivo se ejecuta directamente
- Si es así, ejecutar main() y salir con el código de retorno

**Por qué lo haces:**
Este bloque es estándar en Python. Significa: "Si este archivo se ejecuta directamente (no importado), ejecuta main()".

**Cómo debería funcionar:**
Cuando ejecutas python -m src, el programa ejecuta main() y devuelve un código de salida (0 = éxito, 1 = error).

---

# LECCIÓN 13: MANOS A LA OBRA — FASE 8 (15 minutos)

## Paso 13.1: Ejecutar el programa

**Qué hacer:**
Ejecuta el programa con make run o uv run python -m src.

**Cómo hacerlo:**
Escribe el comando en la terminal.

**Por qué lo haces:**
Necesitas comprobar que el programa funciona correctamente.

**Cómo debería funcionar:**
El programa debería cargar el modelo, procesar todas las preguntas y generar un archivo JSON con los resultados.

---

## Paso 13.2: Comprobar los resultados

**Qué hacer:**
Abre el archivo de salida y comprueba que los resultados son correctos.

**Cómo hacerlo:**
Abre data/output/function_calling_results.json y revisa cada resultado.

**Por qué lo haces:**
Necesitas comprobar que el programa ha elegido las funciones correctas y ha extraído los argumentos correctos.

**Cómo debería funcionar:**
Cada resultado debería tener un prompt, un fn_name y un args con los valores correctos.

---

## Paso 13.3: Depurar errores

**Qué hacer:**
Si hay errores, usa make debug para ejecutar el programa en modo depuración.

**Cómo hacerlo:**
Escribe make debug en la terminal.

**Por qué lo haces:**
El modo depuración te permite ejecutar el programa paso a paso y ver dónde está el error.

**Cómo debería funcionar:**
El programa se detendrá en el error y podrás ver los valores de las variables.

---

# LECCIÓN 14: CREAR tools.py — Función 1: json_reader

## Qué es json_reader?

Es una función que lee un archivo JSON y lo convierte en datos Python. Es la base de todo: sin ella, el programa no puede leer los archivos de entrada.

## Qué necesita hacer:

1. **Recibir una ruta de archivo** (un string o un objeto Path)
2. **Verificar que el archivo exista** — Si no existe, lanzar un error claro
3. **Abrir el archivo** — Usa un context manager (`with open(...)`) para asegurar que el archivo se cierra automáticamente
4. **Cargar el JSON** — Usa la librería `json` para convertir el contenido del archivo en datos Python
5. **Devolver los datos** — El resultado puede ser un diccionario, una lista, o cualquier estructura JSON

## Por qué es importante:

- Sin esta función, el programa no puede leer las funciones disponibles ni las preguntas de prueba
- Es la base para todas las demás funciones de lectura/escritura
- Debe manejar errores claramente: si el archivo no existe o el JSON está mal formado, el usuario debe entender qué pasó

## Cómo debería funcionar:

- Si el archivo existe y el JSON es válido → devuelve los datos
- Si el archivo no existe → lanza un error claro ("Archivo no encontrado: {ruta}")
- Si el JSON está mal formado → lanza un error claro ("JSON inválido en {ruta}: {error}")

## Conceptos clave:

### Context manager
Un context manager es una forma de asegurar que un recurso se cierra automáticamente cuando terminas de usarlo. En Python, se usa la palabra clave `with`:

```python
with open("archivo.txt", "r") as f:
    # Aquí el archivo está abierto
    contenido = f.read()
# Aquí el archivo se ha cerrado automáticamente
```

### Manejo de errores
Es importante manejar errores claramente para que el usuario entendió qué pasó. Usa try/except para capturar errores y lanzar mensajes claros.

---

# LECCIÓN 15: CREAR tools.py — Función 2: load_function_def

## Qué es load_function_def?

Es una función que carga las definiciones de funciones desde un archivo JSON y las valide con Pydantic. Convierte los datos JSON en objetos `FunctionDefinition`.

## Qué necesita hacer:

1. **Recibir una ruta de archivo** — El archivo JSON con las definiciones de funciones
2. **Leer el archivo** — Usa `json_reader` (la función que acabas de crear)
3. **Verificar que sea una lista** — El JSON debe ser un array de funciones
4. **Validar cada función** — Para cada elemento de la lista, crear un objeto `FunctionDefinition`
5. **Manejar errores** — Si alguna definición es inválida, indicar en qué índice falló
6. **Devolver la lista de funciones** — Una lista de objetos `FunctionDefinition`

## Por qué es importante:

- Sin esta función, el programa no puede usar las funciones disponibles
- Valida que las funciones tengan el formato correcto antes de usarlas
- Si hay un error en el JSON, el usuario debe entender exactamente qué está mal y dónde

## Cómo debería funcionar:

- Si el JSON es válido → devuelve una lista de objetos `FunctionDefinition`
- Si el JSON no es una lista → lanza un error claro
- Si alguna función es inválida → lanza un error indicando el índice

## Conceptos clave:

### Verificar tipos
Usa `isinstance(data, list)` para verificar que los datos son una lista antes de iterar.

### Iterar y crear objetos
Para cada elemento en la lista, crea un objeto `FunctionDefinition` y añádelo a una lista.

### Manejar errores en bucles
Usa try/except dentro del bucle para capturar errores e indicar qué elemento falló.

---

# LECCIÓN 16: CREAR tools.py — Función 3: load_prompt

## Qué es load_prompt?

Es una función que carga los prompts de prueba desde un archivo JSON. Los prompts son las preguntas que el sistema debe procesar.

## Qué necesita hacer:

1. **Recibir una ruta de archivo** — El archivo JSON con los prompts
2. **Leer el archivo** — Usa `json_reader`
3. **Verificar que sea una lista** — El JSON debe ser un array de prompts
4. **Extraer el texto de cada prompt** — Cada prompt puede ser un string o un diccionario con un campo "prompt"
5. **Devolver una lista de strings** — Una lista con los textos de los prompts

## Por qué es importante:

- Sin esta función, el programa no puede leer las preguntas de prueba
- Los prompts pueden venir en diferentes formatos (string o diccionario), así que la función debe manejar ambos casos
- Es la última función de lectura que necesitas antes de empezar con la lógica del modelo

## Cómo debería funcionar:

- Si el JSON es válido → devuelve una lista de strings
- Si el JSON no es una lista → lanza un error claro
- Si un prompt es un diccionario → extrae el campo "prompt"
- Si un prompt es un string → lo usa directamente

## Conceptos clave:

### Manejar múltiples formatos
Los prompts pueden venir como strings o diccionarios. Usa `isinstance()` para verificar el tipo y manejar cada caso.

### TestPrompt
Es una clase Pydantic que representa un prompt de prueba. Tiene un campo `prompt` de tipo string.

---

# LECCIÓN 17: CREAR tools.py — Función 4: json_exporter

## Qué es json_exporter?

Es una función que escribe los resultados en un archivo JSON. Es la última función de `tools.py`.

## Qué necesita hacer:

1. **Recibir una lista de objetos `FunctionCall`** — Los resultados del programa
2. **Recibir una ruta de archivo** — Dónde guardar el JSON
3. **Crear la carpeta si no existe** — Si la carpeta `data/output/` no existe, la crea
4. **Convertir los objetos a diccionarios** — Los objetos `FunctionCall` no se pueden escribir directamente a JSON, hay que convertirlos
5. **Escribir el JSON en el archivo** — Usa un context manager para abrir el archivo y escribir

## Por qué es importante:

- Sin esta función, el programa no puede guardar los resultados
- Debe crear la carpeta si no existe, porque si no, fallará al intentar escribir
- Los objetos Pydantic tienen un método para convertirse a diccionarios

## Cómo debería funcionar:

- Si la carpeta no existe → la crea
- Si el archivo se escribe correctamente → no devuelve nada (o devuelve None)
- Si hay un error → lanza un error claro

## Conceptos clave:

### pathlib
`pathlib` es una librería estándar de Python para trabajar con rutas de archivos. Está permitida en el subject porque es parte de la stdlib.

### Crear carpetas
Usa `path.mkdir(parents=True, exist_ok=True)` para crear una carpeta si no existe. El parámetro `parents=True` crea las carpetas padre si no existen, y `exist_ok=True` evita errores si la carpeta ya existe.

### Convertir objetos Pydantic a diccionarios
Los objetos Pydantic tienen un método `.dict()` que los convierte en diccionarios. Esto es necesario porque JSON no puede serializar objetos Pydantic directamente.

### Escribir JSON con formato legible
Usa `json.dump(data, f, indent=4)` para escribir JSON con sangría de 4 espacios. Esto hace que el JSON sea más legible para los humanos.

---

# LECCIÓN 18: CREAR prompt_builder.py

## Qué es prompt_builder.py?

Es un módulo que construye los mensajes (prompts) que le pasamos al modelo. La forma en que le preguntamos al modelo afecta mucho a la respuesta.

## Por qué es importante:

El modelo es como una persona muy inteligente pero que no sabe qué quieres hacer. Si no le das instrucciones claras, no sabrá qué responder.

## Función 1: function_selection (build_function_selection_prompt)

### ¿Qué hace?
Construye un texto que pregunta al modelo qué función quiere usar.

### ¿Qué necesita?
1. La pregunta del usuario (ej: "What is the sum of 2 and 3?")
2. La lista de funciones disponibles (ej: fn_add_numbers, fn_greet, etc.)

### ¿Cómo funciona paso a paso?

1. **Empieza con un rol** — Dile al modelo quién es:
   ```
   "You are a function calling assistant."
   ```

2. **Explica qué debe hacer** — Dile al modelo qué tarea tiene:
   ```
   "Select the most appropriate function for the user's request."
   ```

3. **Lista las funciones disponibles** — Dile al modelo qué funciones puede usar:
   ```
   "Available functions:
   - fn_add_numbers: Add two numbers together and return their sum.
   - fn_greet: Generate a greeting message for a person by name."
   ```

4. **Incluye la pregunta del usuario** — Dile al modelo qué pregunta quiere responder:
   ```
   "User request: What is the sum of 2 and 3?"
   ```

5. **Termina con una indicación** — Dile al modelo qué debe generar:
   ```
   "Function to call: "
   ```

### El resultado final sería:
```
You are a function calling assistant. Select the most appropriate function for the user's request.

Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.

User request: What is the sum of 2 and 3?

Function to call: 
```

### ¿Por qué termina con "Function to call: "?
Porque el modelo "quiere" continuar con el nombre de una función. Es como si le diéramos la primera palabra de una frase y esperáramos que la complete.

## Función 2: arg_extract (build_argument_extraction_prompt)

### ¿Qué hace?
Construye un texto que pregunta al modelo qué valor tiene un parámetro.

### ¿Qué necesita?
1. La pregunta del usuario (ej: "What is the sum of 2 and 3?")
2. La función elegida (ej: fn_add_numbers)
3. El parámetro que quieres extraer (ej: "a")
4. Los argumentos ya extraídos (ej: {"a": 2})

### ¿Cómo funciona paso a paso?

1. **Explica qué función se va a usar** — Dile al modelo qué función ha elegido:
   ```
   "Function: fn_add_numbers - Add two numbers together and return their sum."
   ```

2. **Incluye la pregunta del usuario** — Dile al modelo qué pregunta quiere responder:
   ```
   "User request: What is the sum of 2 and 3?"
   ```

3. **Indica qué parámetro se está extrayendo** — Dile al modelo qué valor necesita:
   ```
   "Extract parameter 'a' (type: number):"
   ```

4. **Incluye los argumentos ya extraídos** — Si ya hay argumentos extraídos, inclúyelos como contexto:
   ```
   "Parameter 'a' (number): already extracted = 2"
   ```

5. **Termina con una indicación** — Dile al modelo qué debe generar:
   ```
   "Value: "
   ```

### El resultado final sería:
```
Function: fn_add_numbers - Add two numbers together and return their sum.
User request: What is the sum of 2 and 3?

Extract parameter 'a' (type: number):
Value: 
```

### ¿Por qué termina con "Value: "?
Porque el modelo "quiere" continuar con el valor del parámetro. Es como si le diéramos la primera palabra de una frase y esperáramos que la complete.

## Conceptos clave:

### Acceder a atributos de objetos
Un objeto `FunctionDefinition` tiene atributos: `name`, `description`, `parameters`, `returns`. Para acceder a ellos, usas el punto:
```python
function.name        # El nombre de la función
function.description # La descripción de la función
```

### f-strings
Los f-strings son strings que pueden incluir variables:
```python
nombre = "fn_add_numbers"
texto = f"- {nombre}: Add two numbers together"
```

### Unir strings con saltos de línea
Usa `"\n".join(lista)` para unir todos los strings de una lista con saltos de línea:
```python
funciones_texto = "\n".join(formated_func)
```

### No es hardcodeado
El modelo es el que decide qué función llamar, no nosotros. La decodificación restringida (que implementaremos después) es lo que garantiza que el modelo solo pueda elegir entre las funciones disponibles.

---

# LECCIÓN 19: CREAR vocab.py — La intención y el qué

## ¿Qué es un vocabulario?

Imagina que tienes un diccionario gigante que dice:

```
"hola" → 1284
"mundo" → 339
"{" → 42
"}" → 43
```

Esto es un **vocabulario**: un mapeo entre palabras (tokens) y números (IDs).

## ¿Por qué lo necesitamos?

El modelo solo entiende números. Cuando le pasas texto, lo convierte a números:

```
"hola mundo" → [1284, 339]
```

El modelo procesa estos números y devuelve puntuaciones para cada número posible. Pero nosotros queremos saber **qué palabra** corresponde a cada número. Para eso necesitamos el vocabulario.

## ¿Cómo se usa en la decodificación restringida?

Imagina que quieres extraer un número. Solo quieres permitir tokens que son dígitos:

```
"0" → ID 10
"1" → ID 11
"2" → ID 12
...
```

Con el vocabulario, puedes buscar todos los tokens que son dígitos y obtener sus IDs. Después, puedes bloquear todos los demás tokens.

## ¿Qué vamos a hacer en vocab.py?

1. **Cargar el vocabulario** — Leer el archivo JSON y crear diccionarios de búsqueda
2. **Detectar el formato** — El JSON puede ser `{"token": id}` o `{"id": token}`
3. **Crear métodos de búsqueda** — Buscar tokens exactos, por prefijo, por caracteres

---

# LECCIÓN 20: CREAR vocab.py — Cargar y detectar el formato

## Paso 1: Cargar el vocabulario

El `vocab.json` del modelo es un diccionario JSON. Puede tener dos formatos:

**Formato A** (clave = token, valor = ID):
```json
{"hola": 1284, "mundo": 339, "{": 42}
```

**Formato B** (clave = ID, valor = token):
```json
{"1284": "hola", "339": "mundo", "42": "{"}
```

## Paso 2: Detectar el formato automáticamente

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

## Paso 3: Crear diccionarios de búsqueda

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

## Conceptos clave:

### Obtener la primera clave de un diccionario
Los diccionarios no se acceden con `[0]`. Para obtener la primera clave:
```python
primera_clave = next(iter(diccionario.keys()))
```

### Crear un diccionario invertido
Para invertir un diccionario (intercambiando claves y valores):
```python
diccionario_invertido = {v: k for k, v in diccionario.items()}
```

### try/except para detectar tipos
Usa try/except para intentar convertir un valor a otro tipo:
```python
try:
    int(valor)
    # Es un número
except (ValueError, TypeError):
    # No es un número
```

---

# LECCIÓN 21: CREAR vocab.py — Método search_exact

## ¿Qué hace?

Busca un token exacto en el diccionario y devuelve su ID.

## Ejemplo:

Si tienes:
```python
self.token_to_id = {"hola": 1284, "mundo": 339}
```

Y buscas `"hola"`, debería devolver `[1284]`.

## ¿Cómo funciona paso a paso?

1. **Recibe un token** — Por ejemplo, `"hola"`
2. **Busca el token en el diccionario** — Usa `.get(token)`
3. **Si existe, devuelve el ID como lista** — `[token_id]`
4. **Si no existe, devuelve una lista vacía** — `[]`

## Conceptos clave:

### Buscar en un diccionario
Usa `.get()` para buscar un valor en un diccionario:
```python
valor = diccionario.get(clave)
# Si la clave no existe, devuelve None
```

### Devolver listas
Aunque solo haya un elemento, devuelve una lista para consistencia:
```python
return [token_id]  # Si existe
return []  # Si no existe
```

---

# LECCIÓN 22: CREAR vocab.py — Método search_prefix

## ¿Qué hace?

Busca todos los tokens que **empiezan por un prefijo** y devuelve sus IDs.

## ¿Por qué es importante?

En la decodificación restringida, necesitas saber qué tokens son válidos en cada paso. Por ejemplo:
- Para extraer un nombre de función, necesitas saber qué tokens empiezan por `"fn_"`
- Para extraer un número, necesitas saber qué tokens son dígitos

## Ejemplo concreto (no relacionado con el proyecto)

Imagina que tienes un diccionario de palabras:

```
"casa" → 1
"coche" → 2
"perro" → 3
"gato" → 4
```

Si buscas todas las palabras que empiezan por `"ca"`, el resultado debería ser:
```
[1, 2]  # casa, coche
```

## ¿Cómo funciona paso a paso?

1. **Recibe un prefijo** — Por ejemplo, `"ca"`
2. **Itera sobre todos los tokens** del diccionario
3. **Verifica si cada token empieza por el prefijo** — Usa el método `.startswith()`
4. **Si coincide, guarda el ID** — Añade el ID a una lista
5. **Devuelve la lista de IDs** — Todos los IDs de tokens que empiezan por el prefijo

## Conceptos clave:

### Verificar si un string empieza por otro
Usa el método `.startswith()`:
```python
token.startswith(prefijo)  # True o False
```

### Iterar sobre un diccionario
Usa `.items()` para obtener clave y valor:
```python
for token, id in diccionario.items():
    # token es la clave, id es el valor
```

---

# LECCIÓN 23: CREAR vocab.py — Método search_characters

## ¿Qué hace?

Busca todos los tokens que están **compuestos solo por ciertos caracteres** y devuelve sus IDs.

## ¿Por qué es importante?

En la decodificación restringida, necesitas saber qué tokens son válidos en cada paso. Por ejemplo:
- Para extraer un número, necesitas saber qué tokens son dígitos (`0-9`)
- Para extraer un número con decimales, necesitas saber qué tokens son dígitos o el punto (`.`)

## Ejemplo concreto (no relacionado con el proyecto)

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

## ¿Cómo funciona paso a paso?

1. **Recibe un string de caracteres permitidos** — Por ejemplo, `"0123456789"`
2. **Itera sobre todos los tokens** del diccionario
3. **Verifica si todos los caracteres del token están en los caracteres permitidos**
4. **Si coincide, guarda el ID** — Añade el ID a una lista
5. **Devuelve la lista de IDs** — Todos los IDs de tokens compuestos solo por esos caracteres

## Conceptos clave:

### Verificar si todos los caracteres están permitidos
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

### Error común: `valid_chars in token`
**Incorrecto:**
```python
if valid_chars in token:  # Verifica si el string completo está en el token
```

**Correcto:**
```python
if all(c in valid_chars for c in token):  # Verifica si todos los caracteres del token están en valid_chars
```

---

# LECCIÓN 24: CREAR constrained_decoder.py — get_next_token_logits

## ¿Qué hace?

Pide al modelo las puntuaciones (logits) para el siguiente token.

## ¿Por qué es importante?

El modelo devuelve puntuaciones para cada token del vocabulario. Necesitas estas puntuaciones para saber qué token es el más probable.

## ¿Qué parámetros recibe?

1. **`model`** — El modelo (objeto Small_LLM_Model)
2. **`input_ids`** — Una lista de IDs de tokens (el contexto actual)

## ¿Qué debe hacer paso a paso?

1. **Llamar al SDK** — Usa `model.get_logits_from_input_ids(input_ids)` para obtener los logits
2. **Convertir a NumPy** — Usa `np.array(logits)` para convertir la lista a un array de NumPy
3. **Devolver el array** — Retorna el array de NumPy

## ¿De dónde salen los input_ids?

Los `input_ids` vienen de **convertir texto a IDs** usando el método `encode` del SDK:

```python
prompt = "What is the sum of 2 and 3?"
input_ids = model.encode(prompt)
# Resultado: [[892, 318, 262, 4771, 286, 16, 290, 17, 30]]
```

## Flujo completo:

```
prompt_builder.py → texto → model.encode() → input_ids → get_next_token_logits() → logits
```

## Conceptos clave:

### encode vs get_logits_from_input_ids
- `encode(text)` — Convierte texto a IDs
- `get_logits_from_input_ids(input_ids)` — Devuelve puntuaciones para el siguiente token

### Convertir a NumPy
El SDK devuelve una lista de floats. Para trabajar con ella fácilmente, conviértela a un array de NumPy:
```python
logits_array = np.array(logits)
```

---

# LECCIÓN 25: CREAR constrained_decoder.py — apply_mask

## ¿Qué hace?

Pone **-infinito** a los tokens que no son válidos. Así el modelo no puede elegirlos nunca.

## ¿Por qué es importante?

El modelo elige el token con la puntuación más alta. Si pones -infinito a los tokens inválidos, el modelo no puede elegirlos nunca.

## Ejemplo concreto (no relacionado con el proyecto)

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

## ¿Qué parámetros recibe?

1. **`logits`** — Array de NumPy con las puntuaciones
2. **`valid_ids`** — Lista de IDs válidos

## ¿Qué debe hacer paso a paso?

1. **Crear un array nuevo lleno de -infinito** — Todos los tokens empiezan bloqueados
2. **Para cada ID válido, copiar su puntuación original** — Los tokens válidos se desbloquean
3. **Devolver el array enmascarado**

## Conceptos clave:

### Crear un array lleno de -infinito
Usa `np.full()` para crear un array lleno de un valor:
```python
masked = np.full(len(logits), -np.inf)
```

### Copiar puntuaciones de tokens válidos
Itera sobre los IDs válidos y copia sus puntuaciones:
```python
for valid_id in valid_ids:
    masked[valid_id] = logits[valid_id]
```

### Error común: usar el índice incorrecto
**Incorrecto:**
```python
masked[valid_id] = logits[valid_ids]  # ❌ valid_ids es una lista
```

**Correcto:**
```python
masked[valid_id] = logits[valid_id]  # ✅ valid_id es un número
```

---

# LECCIÓN 26: CREAR constrained_decoder.py — select_best_token

## ¿Qué hace?

Elige el token con la **puntuación más alta** del array enmascarado.

## ¿Por qué es importante?

Después de aplicar la máscara, los tokens inválidos tienen -infinito. El modelo debe elegir el token con la puntuación más alta entre los válidos.

## Ejemplo concreto (no relacionado con el proyecto)

Imagina que tienes puntuaciones para 5 tokens:

```
Token "0" → puntuación 5.2
Token "1" → puntuación 3.1
Token "a" → -infinito (bloqueado)
Token "b" → -infinito (bloqueado)
Token "2" → puntuación 4.0
```

La función debe devolver el ID del token con la puntuación más alta: `0` (puntuación 5.2).

## ¿Qué parámetros recibe?

1. **`masked_logits`** — Array de NumPy con las puntuaciones enmascaradas

## ¿Qué debe hacer paso a paso?

1. **Encontrar el token con la puntuación más alta** — Usa `np.argmax()`
2. **Devolver su ID** — El ID del token con la puntuación más alta

## Conceptos clave:

### np.argmax()
`np.argmax()` devuelve el **índice** del valor más alto en un array:
```python
masked_logits = np.array([5.2, 3.1, -np.inf, -np.inf, 4.0])
best_token_id = np.argmax(masked_logits)
# Resultado: 0 (el índice del valor 5.2)
```

---

# LECCIÓN 27: CREAR constrained_decoder.py — JSONGenerator

## ¿Qué es JSONGenerator?

Es la clase que coordina todo el proceso de decodificación restringida. Usa las tres funciones que ya has creado (`get_next_token_logits`, `apply_mask`, `select_best_token`) para generar texto token a token, asegurándose de que solo se eligen tokens válidos.

## ¿Qué métodos necesita?

1. **`select_function_name`** — Elige el nombre de una función usando decodificación restringida
2. **`extract_number`** — Extrae un número usando decodificación restringida
3. **`extract_string`** — Extrae un string usando decodificación restringida
4. **`extract_boolean`** — Extrae un boolean usando decodificación restringida

## ¿Qué parámetros recibe el constructor?

1. **`model`** — El modelo (objeto Small_LLM_Model)
2. **`vocab`** — El vocabulario (objeto VocabIndex)

## ¿Qué hace el constructor?

Guarda el modelo y el vocabulario como atributos de la clase:
```python
def __init__(self, model: Small_LLM_Model, vocab: VocabIndex):
    self.model = model
    self.vocab = vocab
```

## ¿Dónde se crea el vocabulario?

El vocabulario se crea **fuera** de la clase y se pasa como parámetro:
```python
model = Small_LLM_Model()
vocab_path = model.get_path_to_vocab_file()
vocab = VocabIndex(vocab_path)
generator = JSONGenerator(model, vocab)
```

## ¿Por qué no crear el vocabulario dentro de la clase?

Porque:
1. **Es más fácil de probar** — Puedes pasar un vocabulario falso en tests
2. **Es más flexible** — Puedes cambiar el vocabulario sin cambiar la clase
3. **Es más limpio** — La clase no necesita saber cómo se crea el vocabulario

## Conceptos clave:

### Métodos de una clase necesitan `self`
Todos los métodos de una clase necesitan `self` como primer parámetro:
```python
def select_function_name(self, prompt_ids: list, function_list: list):
    # self.model y self.vocab están disponibles aquí
    pass
```

### No pasar model como parámetro si ya está en self
**Incorrecto:**
```python
def select_function_name(self, model: Small_LLM_Model, prompt_ids: list):
    # model es redundante, ya tenemos self.model
```

**Correcto:**
```python
def select_function_name(self, prompt_ids: list):
    # Usa self.model
```

### Llamar a funciones fuera de la clase
**Incorrecto:**
```python
class JSONGenerator:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    logs = get_next_token_logits(model)  # ❌ Está dentro de la clase pero fuera de cualquier método
```

**Correcto:**
```python
class JSONGenerator:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

# Fuera de la clase:
model = Small_LLM_Model()
vocab = VocabIndex(vocab_path)
generator = JSONGenerator(model, vocab)
```

---

# LECCIÓN 28: CREAR constrained_decoder.py — select_function_name

## ¿Qué hace?

Elige el nombre de una función usando decodificación restringida. Compara todos los nombres de funciones disponibles y elige el que tiene la puntuación más alta.

## ¿Qué parámetros recibe?

1. **`prompt_ids`** — Lista de IDs del prompt (el texto convertido a IDs)
2. **`function_names`** — Lista de nombres de funciones disponibles (ej: ["fn_add_numbers", "fn_greet"])

## ¿Cómo funciona paso a paso?

1. **Convierte cada nombre de función a IDs** — Usa `self.model.encode(function_name).flatten().tolist()`
2. **Para cada nombre, calcula una puntuación** — Usa `get_next_token_logits` y sumar los logits de cada token
3. **Elige el nombre con la puntuación más alta** — El nombre más probable

## Ejemplo concreto (no relacionado con el proyecto):

Imagina que tienes dos funciones:
- `sumar` → IDs [1, 2, 3]
- `restar` → IDs [4, 5, 6]

El modelo devuelve logits para el siguiente token. Para cada función, calculas la puntuación total de sus tokens:

```
sumar: 0.7 + 0.6 + 0.8 = 2.1
restar: 0.3 + 0.2 + 0.1 = 0.6
```

La función con la puntuación más alta es `sumar`.

## Conceptos clave:

### Convertir texto a IDs
Usa `model.encode(text).flatten().tolist()` para obtener una lista de IDs:
```python
tensor_ids = self.model.encode(fun_name)
ids = tensor_ids.flatten().tolist()
```

### Calcular la puntuación total de una secuencia
Suma los logits de cada token:
```python
score = 0
for token_id in ids:
    score += logits[token_id]
```

### Elegir el nombre con la puntuación más alta
Compara las puntuaciones y devuelve el nombre con la puntuación más alta:
```python
best_name = max(scores, key=scores.get)
```

---

# LECCIÓN 29: CREAR constrained_decoder.py — extract_number

## ¿Qué hace?

Extrae un número usando decodificación restringida. El modelo genera el número token a token, y en cada paso solo puede elegir tokens válidos (dígitos, punto decimal, signo negativo).

## ¿Por qué es importante?

Los números son uno de los tipos de datos más comunes en las funciones. Sin esta función, el programa no puede extraer números de las preguntas.

## ¿Qué parámetros recibe?

1. **`prompt_ids`** — Lista de IDs del prompt (el texto convertido a IDs)

## ¿Qué variables locales necesita?

1. **`digits`** — IDs de tokens que son dígitos (0-9)
2. **`dot`** — IDs de tokens que son punto decimal
3. **`minus`** — IDs de tokens que son signo negativo
4. **`terminators`** — IDs de tokens que indican el fin del número
5. **`valid_ids`** — Lista combinada de todos los tokens válidos
6. **`tokens`** — Lista vacía para acumular los tokens generados

## ¿Cómo funciona paso a paso?

1. **Buscar tokens válidos** — Usa `self.vocab.search_characters()` para encontrar dígitos, punto, signo y terminadores
2. **Combinar tokens válidos** — Crea `valid_ids` combinando todas las listas
3. **Crear una lista vacía** — `tokens = []` para acumular los tokens generados
4. **Bucle principal:**
   - Obtener logits del modelo
   - Aplicar máscara (solo tokens válidos)
   - Elegir el token con la puntuación más alta
   - Si es terminador → parar
   - Si no → acumular y actualizar prompt_ids
5. **Decodear tokens** — Convertir los IDs a un string
6. **Convertir a float** — Devolver el número como float

## Flujo de datos:

```
prompt_ids → get_next_token_logits → logits
logits → apply_mask → masked
masked → select_best_token → selected
selected → ¿es terminador? → sí: parar / no: acumular y actualizar prompt_ids
tokens → decode → string → float → return
```

## Conceptos clave:

### ¿Por qué incluir terminadores en valid_ids?

Porque el modelo **debe poder generarlos**. Si no los incluyes, el modelo nunca podría terminar y el bucle sería infinito.

### ¿Por qué actualizar prompt_ids?

Porque el modelo necesita saber qué tokens ya ha generado para predecir el siguiente token correctamente.

### ¿Por qué decodear los tokens?

Porque los tokens son IDs numéricos, no strings. Necesitas convertirlos a texto antes de convertir a float.

## Errores comunes:

1. **`valid_ids` no definido** — No creaste la lista antes de usarla
2. **`valid_ids` sin terminadores** — El modelo no podría terminar
3. **No actualizar prompt_ids** — El modelo no sabría qué tokens ya ha generado
4. **No decodear tokens** — Los IDs numéricos no se pueden convertir a float directamente

---

# LECCIÓN 30: Máquina de estados (FSM)

## ¿Qué es una máquina de estados?

Es una forma de saber **en qué estado estamos** y **qué tokens son válidos** en cada estado.

## ¿Por qué es importante?

Sin la máquina de estados, no sabrías qué tokens son válidos en cada paso. Por ejemplo, después de un punto decimal, solo pueden venir dígitos, no letras.

## Ejemplo concreto (no relacionado con el proyecto):

Imagina que estás extrayendo un número. La máquina de estados sería:

```
Estado 0: Esperando dígito o signo negativo
  → Si es dígitos: pasar al estado 1
  → Si es "-": pasar al estado 2

Estado 1: Esperando dígito, punto o terminador
  → Si es dígitos: quedarse en estado 1
  → Si es ".": pasar al estado 3
  → Si es terminador: terminar

Estado 2: Esperando dígito (después de "-")
  → Si es dígitos: pasar al estado 1

Estado 3: Esperando dígito (después de ".")
  → Si es dígitos: quedarse en estado 3
  → Si es terminador: terminar
```

## ¿Cómo se implementa en Python?

Usa un bucle `while` con un `break` cuando se encuentre un terminador:

```python
while True:
    # Obtener logits, aplicar máscara, elegir token
    if token_es_terminador:
        break
    # Acumular token y actualizar estado
```

---

# LECCIÓN 31: CREAR constrained_decoder.py — extract_string

## ¿Qué hace?

Extrae un string usando decodificación restringida. El modelo genera el string token a token, y en cada paso solo puede elegir tokens válidos.

## ¿Por qué es importante?

Los strings son uno de los tipos de datos más comunes en las funciones. Sin esta función, el programa no puede extraer strings de las preguntas.

## ¿Qué parámetros recibe?

1. **`prompt_ids`** — Lista de IDs del prompt (el texto convertido a IDs)

## ¿Qué variables locales necesita?

1. **`terminators`** — IDs de tokens que indican el fin del string (`"`, `\n`)
2. **`not_valids`** — IDs de tokens que contienen caracteres que rompen JSON (`{`, `}`, `[`, `]`)
3. **`valid_ids`** — Lista de IDs de tokens válidos (todos excepto los que rompen JSON)
4. **`tokens`** — Lista vacía para acumular los tokens generados

## ¿Qué tokens rompen la estructura JSON?

Los tokens que contienen `{`, `}`, `[`, `]` rompen la estructura JSON. El modelo no debe poder generarlos mientras extrae un string.

## ¿Qué tokens son terminadores?

Los tokens que indican el fin del string son `"` (comillas) y `\n` (salto de línea).

## ¿Cómo funciona paso a paso?

1. **Buscar tokens terminadores** — Usa `self.vocab.search_characters('"\n')`
2. **Buscar tokens válidos** — Todos los tokens que no contienen `{`, `}`, `[`, `]`
3. **Crear una lista vacía** — `tokens = []` para acumular los tokens generados
4. **Bucle principal:**
   - Obtener logits del modelo
   - Aplicar máscara (solo tokens válidos)
   - Elegir el token con la puntuación más alta
   - Si es terminador → parar
   - Si no → acumular y actualizar prompt_ids
5. **Decodear tokens** — Convertir los IDs a un string
6. **Devolver el string**

## Flujo de datos:

```
prompt_ids → get_next_token_logits → logits
logits → apply_mask → masked
masked → select_best_token → selected
selected → ¿es terminador? → sí: parar / no: acumular y actualizar prompt_ids
tokens → decode → string → return
```

## Conceptos clave:

### Buscar tokens que no contienen ciertos caracteres

Itera sobre el vocabulario y verifica si el token contiene caracteres inválidos:

```python
for token, token_id in self.vocab.token_to_id.items():
    if not any(char in token for char in not_valids):
        valid_ids.append(token_id)
```

### Error común: search_characters con múltiples argumentos

**Incorrecto:**
```python
not_valids = self.vocab.search_characters("{", "}", "[", "]")  # ❌
```

**Correcto:**
```python
not_valids = self.vocab.search_characters("{}[]")  # ✅
```

### Error común: usar id_to_token en vez de token_to_id

**Incorrecto:**
```python
for token, token_id in self.vocab.id_to_token.items():  # ❌ id_to_token mapea ID → token
```

**Correcto:**
```python
for token, token_id in self.vocab.token_to_id.items():  # ✅ token_to_id mapea token → ID
```

---

# LECCIÓN 32: CREAR constrained_decoder.py — extract_boolean

## ¿Qué hace?

Extrae un boolean (`true` o `false`) usando decodificación restringida.

## ¿Por qué es importante?

Los booleans son un tipo de datos común en las funciones. Sin esta función, el programa no puede extraer booleans de las preguntas.

## ¿Qué parámetros recibe?

1. **`prompt_ids`** — Lista de IDs del prompt (el texto convertido a IDs)

## ¿Qué variables locales necesita?

1. **`true_id`** — IDs del token `true`
2. **`false_id`** — IDs del token `false`
3. **`valid_ids`** — Lista combinada de `true_id` y `false_id`

## ¿Cómo funciona paso a paso?

1. **Buscar los IDs de `true` y `false`** — Usa `self.vocab.search_exact("true")` y `self.vocab.search_exact("false")`
2. **Combinar en `valid_ids`** — `valid_ids = true_id + false_id`
3. **Obtener logits del modelo** — `get_next_token_logits(self.model, prompt_ids)`
4. **Aplicar máscara** — `apply_mask(bool_logits, valid_ids)`
5. **Elegir token** — `select_best_token(masked)`
6. **Devolver boolean** — `return selected in true_id`

## Conceptos clave:

### Buscar tokens exactos

Usa `search_exact()` para buscar un token exacto:

```python
true_id = self.vocab.search_exact("true")
false_id = self.vocab.search_exact("false")
```

### Error común: search_characters en vez de search_exact

**Incorrecto:**
```python
true_id = self.vocab.search_characters("true")  # ❌ Busca tokens compuestos solo por t,r,u,e
```

**Correcto:**
```python
true_id = self.vocab.search_exact("true")  # ✅ Busca el token exacto "true"
```

### Comparar puntuaciones

Para saber si `true` o `false` tiene mayor puntuación, usa `select_best_token` y verifica cuál fue elegido:

```python
selected = select_best_token(masked)
return selected in true_id  # True si se eligió "true", False si se eligió "false"
```

---

# PASOS LÓGICOS DEL PROYECTO

## Paso 1: Preparar el entorno

1. Crear la estructura de carpetas
2. Crear `pyproject.toml` con dependencias
3. Crear `Makefile` con atajos
4. Copiar `llm_sdk` en la raíz
5. Crear archivos de datos de entrada

## Paso 2: Crear los modelos de datos (models.py)

1. Crear `ParameterDefinition` — Define un parámetro de función
2. Crear `FunctionDefinition` — Define una función completa
3. Crear `FunctionCall` — Define el resultado final
4. Crear `TestPrompt` — Define un prompt de prueba

## Paso 3: Crear las herramientas de entrada/salida (tools.py)

1. Crear `json_reader` — Lee archivos JSON
2. Crear `load_function_def` — Carga y valida funciones
3. Crear `load_prompt` — Carga prompts de prueba
4. Crear `json_exporter` — Escribe resultados en JSON

## Paso 4: Crear el constructor de prompts (prompt_builder.py)

1. Crear `function_selection` — Construye prompt para elegir función
2. Crear `arg_extract` — Construye prompt para extraer argumentos

## Paso 5: Crear el vocabulario (vocab.py)

1. Crear `VocabIndex` — Carga el vocabulario
2. Crear `search_exact` — Busca tokens exactos
3. Crear `search_prefix` — Busca tokens por prefijo
4. Crear `search_characters` — Busca tokens por caracteres

## Paso 6: Crear el decodificador restringido (constrained_decoder.py)

1. Crear `get_next_token_logits` — Obtiene logits del modelo
2. Crear `apply_mask` — Aplica máscara a los logits
3. Crear `select_best_token` — Elige el token con la puntuación más alta
4. Crear `JSONGenerator` — Clase que coordina todo
5. Crear `select_function_name` — Elige el nombre de una función
6. Crear `extract_number` — Extrae un número
7. Crear `extract_string` — Extrae un string
8. Crear `extract_boolean` — Extrae un boolean

## Paso 7: Crear el orquestador (function_caller.py)

1. Crear `FunctionCaller` — Clase que coordina todo
2. Crear `resolve` — Dada una pregunta, elige función y extrae argumentos

## Paso 8: Actualizar el punto de entrada (__main__.py)

1. Importar todas las funciones y clases necesarias
2. Cargar archivos de entrada
3. Cargar el modelo
4. Crear el FunctionCaller
5. Procesar cada prompt
6. Escribir resultados

## Paso 9: Probar y depurar

1. Ejecutar el programa
2. Comprobar los resultados
3. Depurar errores

---

# LECCIÓN 33: CREAR function_caller.py — Explicación completa

## ¿Qué es FunctionCaller?

Es el **orquestador** que coordina todo el proceso. Es como el director de orquesta: no toca ningún instrumento, pero sabe cuándo debe entrar cada uno.

## ¿Por qué es importante?

Sin FunctionCaller, los otros módulos (prompt_builder, constrained_decoder, tools) no pueden trabajar juntos. FunctionCaller es el que los une y hace que funcionen como un equipo.

## ¿Qué hace el constructor?

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

## ¿Qué hace resolve?

1. **Construir prompt de selección** — Usa `function_selection(prompt, functions_list)`
2. **Convertir prompt a IDs** — Usa `self.model.encode(prompt_txt).flatten().tolist()`
3. **Elegir función** — Usa `self.generator.select_function_name(prompt_ids, funct_names)`
4. **Para cada parámetro:**
   - Construir prompt de extracción con `arg_extract`
   - Convertir prompt a IDs
   - Extraer valor según el tipo (number, string, boolean)
   - Guardar el valor en `extracted_args`
5. **Ensamblar resultado** — Crea un `FunctionCall` con la pregunta, función y argumentos

## Flujo de datos completo:

```
prompt (str)
    ↓
function_selection(prompt, functions_list) → prompt_txt (str)
    ↓
model.encode(prompt_txt) → prompt_ids (list[int])
    ↓
generator.select_function_name(prompt_ids, funct_names) → sel_funct (FunctionDefinition)
    ↓
for param_name, param_def in sel_funct.parameters.items():
    ↓
    arg_extract(sel_funct, prompt, param_name, extracted_args) → arg_prompt (str)
    ↓
    model.encode(arg_prompt) → arg_prompt_ids (list[int])
    ↓
    generator.extract_number/string/boolean(arg_prompt_ids) → value
    ↓
    extracted_args[param_name] = value
    ↓
FunctionCall(prompt=prompt, fn_name=sel_funct.name, args=extracted_args)
```

## De dónde viene cada dato:

| Dato | De dónde viene |
|------|----------------|
| `prompt` | La pregunta del usuario (string) |
| `functions_list` | El archivo `function_definitions.json` (lista de FunctionDefinition) |
| `prompt_txt` | `function_selection()` (string) |
| `prompt_ids` | `model.encode()` (lista de IDs) |
| `funct_names` | `fun.name` de cada FunctionDefinition (lista de strings) |
| `sel_funct` | `generator.select_function_name()` (FunctionDefinition) |
| `arg_prompt` | `arg_extract()` (string) |
| `arg_prompt_ids` | `model.encode()` (lista de IDs) |
| `value` | `generator.extract_number/string/boolean()` (float, string, boolean) |
| `extracted_args` | Diccionario que acumula los valores extraídos |
| `FunctionCall` | El resultado final con prompt, fn_name y args |

## Conceptos clave:

### FunctionCaller es el orquestador

No hace nada por sí solo. Solo coordina los otros módulos:
- **prompt_builder** — Construye los prompts
- **constrained_decoder** — Genera las respuestas
- **tools** — Lee y escribe archivos

### El flujo es secuencial

1. Primero se elige la función
2. Después se extraen los argumentos uno por uno
3. Finalmente se ensambla el resultado

### Los datos fluyen en una dirección

Los datos fluyen de arriba a abajo:
- Entrada: `prompt` y `functions_list`
- Proceso: prompts → IDs → función → argumentos
- Salida: `FunctionCall`

## Errores comunes:

1. **No guardar los valores extraídos** — `extracted_args[param_name] = value` debe estar dentro del bucle
2. **Usar el prompt equivocado** — Usa `prompt` (la pregunta original), no `prompt_txt` (el prompt de selección)
3. **No manejar tipos desconocidos** — Añade un `else: value = None` para tipos no reconocidos

---

# RESUMEN DE LAS FASES

1. **Fase 0**: Preparar el entorno (carpetas, pyproject.toml, Makefile, llm_sdk, datos)
2. **Fase 1**: Crear los modelos de datos (models.py)
3. **Fase 2**: Crear el vocabulario (vocab.py)
4. **Fase 3**: Crear el decodificador restringido (constrained_decoder.py)
5. **Fase 4**: Crear el constructor de prompts (prompt_builder.py)
6. **Fase 5**: Crear el orquestador (function_caller.py)
7. **Fase 6**: Crear las herramientas de entrada/salida (tools.py)
8. **Fase 7**: Crear el punto de entrada (__main__.py)
9. **Fase 8**: Probar y depurar

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
