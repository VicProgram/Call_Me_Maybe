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
Necesitas leer los archivos de entrada (funciones y preguntas) para que el programa pueda procesarlos.

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
Necesitas validar que las funciones tengan el formato correcto antes de usarlas.

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
Necesitas leer las preguntas de prueba para que el programa pueda procesarlas.

**Cómo debería funcionar:**
Cuando llamas a la función, devuelve una lista of strings.

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
Necesitas guardar los resultados para que el usuario pueda verlos.

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
