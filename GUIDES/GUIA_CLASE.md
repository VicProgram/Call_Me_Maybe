# GUÍA UNIVERSITARIA: Call Me Maybe

## Una explicación completa, como si estuvieras en clase

---

# LECCIÓN 1: EL PROBLEMA QUE VAMOS A RESOLVER

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

# LECCIÓN 2: CÓMO FUNCIONA UN MODELO DE LENGUAJE POR DENTRO

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

# LECCIÓN 3: LA DECODIFICACIÓN RESTRINGIDA

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

# LECCIÓN 4: ARQUITECTURA DEL PROYECTO

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

# LECCIÓN 5: EXPLICACIÓN DETALLADA DE CADA ARCHIVO

## 5.1. models.py — Las plantillas de datos

### ¿Qué hace?

Define las **formas** que tienen que tener los datos. Como esos moldes de galletas: si el dato no encaja en el molde, el programa se da cuenta y lanza un error.

### ¿Por qué Pydantic?

Pydantic es una librería que valida datos automáticamente. Imagina que el archivo de funciones tiene un error: un parámetro que debería ser un texto pero es un número.

Cuando Pydantic intente meter eso en la plantilla, verá que no encaja y lanzará un error clarísimo: "el campo 'type' debe ser de tipo str, no int".

Sin Pydantic, ese error pasaría desapercibido y el programa fallaría más tarde de forma misteriosa.

### Las 3 plantillas

**ParameterDefinition** — Define UN parámetro de una función. Cada parámetro solo tiene un campo: type, que puede ser "number", "string", etc.

**FunctionDefinition** — Define UNA función completa. Tiene:
- name: el nombre (ej: "fn_add_numbers")
- description: qué hace (ej: "Add two numbers...")
- parameters: un diccionario de parámetros
- returns: qué devuelve la función

**FunctionCall** — El resultado final. Representa exactamente lo que queremos generar:
- prompt: la pregunta original
- fn_name: el nombre de la función elegida
- args: los argumentos extraídos

## 5.2. vocab.py — El diccionario token ↔ ID

### ¿Qué hace?

Carga el archivo vocab.json del modelo (que puede tener 150.000 entradas) y construye un índice de búsqueda en ambas direcciones:

1. **ID → Token**: dado un ID numérico, ¿qué texto representa?
2. **Token → IDs**: dado un texto, ¿qué IDs lo representan?

### El problema del formato

El archivo vocab.json puede venir en dos formatos distintos:

**Formato A** (clave = token, valor = ID):
```json
{"hello": 334, "world": 335, "{": 42}
```

**Formato B** (clave = ID, valor = token):
```json
{"334": "hello", "335": "world", "42": "{"}
```

El código detecta automáticamente cuál es. Mira la primera clave del archivo. Si parece un número (como "334"), usa el formato B. Si parece texto (como "hello"), usa el formato A.

### Métodos de búsqueda

- get_ids_exact(":") → busca tokens que sean exactamente ":"
- get_ids_chars("0123456789") → busca tokens compuestos solo por esos caracteres (dígitos)
- get_ids_start_with("fn_") → busca tokens que empiecen por "fn_"

## 5.3. constrained_decoder.py — EL CORAZÓN DEL PROYECTO

### ¿Qué hace?

Este archivo implementa la **decodificación restringida**. Es la parte más importante y la que realmente evalúa el proyecto.

### Las funciones auxiliares

**get_next_token_logits(model, input_ids)**

Le pide al modelo las puntuaciones (logits) para el siguiente token. El modelo recibe la lista de IDs y devuelve 150.000 números (uno por cada token de su vocabulario).

**apply_mask(logits, valid_ids)**

Esta función ES la decodificación restringida en su forma más pura. Lo que hace:
1. Crea un array nuevo lleno de -infinito (menos infinito)
2. Para cada ID que SÍ es válido, copia su logit original a esa posición
3. Los tokens inválidos se quedan con -infinito

Resultado: los tokens inválidos **nunca** serán elegidos, porque -infinito es la puntuación más baja posible.

**select_best_token(masked_logits)**

Elige el token con la puntuación más alta del array enmascarado. Si todos son -infinito (caso de error), devuelve None.

### La clase JSONGenerator

Esta clase contiene la lógica de generación con restricciones. Tiene los siguientes métodos:

#### select_function_name(prompt_ids, candidate_names)

**Escenario**: Tenemos una pregunta como "What is the sum of 2 and 3?" y 5 funciones disponible.

**Proceso**:
1. Pasamos el prompt "Function to call: " al modelo y obtenemos los logits del siguiente token
2. Convertimos los logits a log-probabilidades (log-softmax)
3. Para cada nombre de función candidato:
   a. Codificamos el nombre a IDs de tokens
   b. Calculamos la probabilidad total de esa secuencia de tokens
4. Elegimos el nombre con mayor probabilidad acumulada

**¿Por qué sumamos log-probabilidades en lugar de multiplicar probabilidades?**

Porque las probabilidades son números muy pequeños (entre 0 y 1). Multiplicar muchos números pequeños da un número diminutivo que el ordenador no puede manejar bien. En cambio, sumar logaritmos es equivalente a multiplicar probabilidades, pero es mucho más estable numéricamente.

**Ejemplo concreto**:

Imagina que el modelo asigna estas probabilidades al primer token de cada nombre:

| Nombre | 1er token | Probabilidad |
|--------|-----------|-------------|
| fn_add_numbers | "fn" | 0.7 |
| fn_greet | "fn" | 0.7 |
| fn_reverse_string | "fn" | 0.7 |
| fn_get_square_root | "fn" | 0.7 |
| fn_substitute... | "fn" | 0.7 |

Todos empiezan igual, así que necesitamos ver los tokens siguientes. El método _score_sequence hace forward passes adicionales para tokens posteriores del nombre.

Al final, si la pregunta es sobre sumar números, fn_add_numbers tendrá la puntuación más alta.

#### _log_softmax(logits)

Convierte logits en bruto a log-probabilidades. La implementación es **numéricamente estable**: resta el máximo antes de hacer exponenciales para evitar que los números se disparen.

#### extract_number(prompt_ids)

**Escenario**: Tenemos que extraer "40" para el parámetro 'a'.

**Proceso**:
1. Buscamos todos los tokens que son dígitos (0-9), el punto decimal (.) y el signo negativo (-)
2. También permitimos "terminadores": coma, llave de cierre, salto de línea, espacio (para detectar que el número ha terminado)
3. En cada paso:
   - Si no hemos acumulado nada: permitimos dígitos o signo negativo
   - Si hemos acumulado solo "-": permitimos solo dígitos
   - Si ya hay punto decimal: permitimos solo dígitos
   - Si no: permitimos dígitos o punto decimal
4. Aplicamos la máscara (solo tokens válidos)
5. El modelo elige el mejor token
6. Si es un terminador: paramos y devolvemos el número acumulado
7. Si no: acumulamos el token y repetimos

**Ejemplo paso a paso para extraer "40.0"**:

Paso 1: prompt = "Extract parameter 'a' (type: number):\nValue: "
- Tokens válidos: ["0", "1", ..., "9", "-"]
- El modelo elige: "4"
- Acumulado: "4"

Paso 2: Tokens válidos: ["0", "1", ..., "9", "."]
- El modelo elige: "0"
- Acumulado: "40"

Paso 3: Tokens válidos: ["0", "1", ..., "9", "."]
- El modelo elige: "."
- Acumulado: "40."

Paso 4: Tokens válidos: ["0", "1", ..., "9"]
- El modelo elige: "0"
- Acumulado: "40.0"

Paso 5: Tokens válidos: ["0", "1", ..., "9", ".", terminadores]
- El modelo elige: "," (terminador)
- Se para, devuelve float("40.0") = 40.0

#### extract_string(prompt_ids)

**Escenario**: Tenemos que extraer "shrek" para el parámetro 'name'.

**Proceso**:
1. Creamos una lista de tokens válidos: TODOS excepto los que rompen la estructura JSON ({, }, [, ])
2. Definimos tokens que indican el fin del string (\n, ", ")
3. En cada paso:
   - Aplicamos la máscara
   - El modelo elige un token
   - Si termina en fin-de-string: paramos
   - Si no: acumulamos

**Limitación**: Este método es el más simple. Para valores string que contienen comillas o caracteres especiales, puede fallar. Pero para los casos de prueba del proyecto (nombres, palabras simples) funciona bien.

#### extract_boolean(prompt_ids)

**Escenario**: Tenemos que decidir entre "true" o "false".

**Proceso**:
1. Obtenemos los logits y calculamos log-probabilidades
2. Codificamos "true" y "false" a IDs
3. Puntuamos la secuencia completa de "true" y "false"
4. Devolvemos True si "true" tiene mayor puntuación

## 5.4. prompt_builder.py — Construir los mensajes para la IA

### ¿Qué hace?

Construye los textos (prompts) que le pasamos al modelo. La forma en que le preguntamos al modelo afecta MUCHO a la respuesta.

### build_function_selection_prompt

Construye un prompt como este:

```
You are a function calling assistant. Select the most appropriate function.

Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- fn_reverse_string: Reverse a string and return the reversed result.
- fn_get_square_root: Calculate the square root of a number.
- fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.

User request: What is the sum of 2 and 3?

Function to call: 
```

Fíjate que el prompt termina con "Function to call: ". Esto hace que el modelo "quiera" continuar con el nombre de una función. Es como si le diéramos la primera palabra de una frase y esperáramos que la complete.

### build_argument_extraction_prompt

Construye un prompt como este:

```
Function: fn_add_numbers - Add two numbers together and return their sum.
User request: What is the sum of 2 and 3?

Extract parameter 'a' (type: number):
Value: 
```

Y si ya hemos extraído algún parámetro, lo incluimos como contexto:

```
Function: fn_add_numbers - Add two numbers together and return their sum.
User request: What is the sum of 2 and 3?
Parameter 'a' (number): already extracted = 40.0

Extract parameter 'b' (type: number):
Value: 
```

Esto ayuda al modelo a saber qué valor se espera y a mantener coherencia.

## 5.5. function_caller.py — El orquestador

### ¿Qué hace?

Coordina todo el proceso. Es como el director de orquesta: no toca ningún instrumento, pero sabe cuándo debe entrar cada uno.

### ¿Por qué dos fases?

Podríamos intentar que el modelo generara todo el JSON de golpe. Pero eso es mucho más complejo (habría que trackear el estado del JSON token a token) y más propenso a errores con un modelo pequeño.

Dividir en fases:
1. La función se elige por puntuación (solo entre nombres válidos)
2. Cada argumento se extrae por separado con restricciones específicas según su tipo

Esto hace que cada paso sea simple y que el modelo solo tenga que hacer una cosa a la vez.

## 5.6. tools.py — Leer y escribir archivos

### ¿Qué hace?

Operaciones de entrada/salida: leer los archivos JSON y escribir los resultados.

### json_reader(path)

Lee un archivo JSON y lo convierte en datos Python. Si el archivo no existe, lanza error. Si el JSON está mal formado, también.

### load_function_def(path)

Carga las definiciones de funciones y las valida con Pydantic. Lee el JSON, verifica que sea una lista, y para cada elemento crea un objeto FunctionDefinition. Si alguna definición es inválida, indica en qué índice falló.

### load_prompt(path)

Carga los prompts de prueba. El formato esperado es una lista de objetos con un campo "prompt". También acepta strings directamente.

### json_exporter(results, path)

Escribe los resultados en un archivo JSON. Crea la carpeta si no existe.

## 5.7. __main__.py — El punto de entrada

### ¿Qué hace?

Es el archivo que se ejecuta cuando escribes "python -m src" (el "-m src" busca src/__main__.py).

### parse_args()

Define los argumentos que acepta el programa:
- --input: Directorio de entrada (por defecto: data/input)
- --output: Archivo de salida (por defecto: data/output/function_calling_results.json)

### main()

El flujo completo:
1. Parsear argumentos de línea de comandos
2. Determinar las rutas de los archivos de entrada
3. Cargar archivos (si falla, salir con código 1)
4. Cargar el modelo Qwen3-0.6B
5. Crear el FunctionCaller
6. Para cada prompt:
   - Extraer el texto del prompt (puede venir como string o como diccionario)
   - Llamar a caller.resolve()
   - Si falla: contar error y continuar
7. Escribir los resultados
8. Mostrar resumen
9. Devolver 0 si todo fue bien, 1 si hubo errores

### El bloque if __name__ == "__main__"

Este bloque es estándar en Python. Significa: "Si este archivo se ejecuta directamente (no importado), ejecuta main()".

El sys.exit() hace que el programa devuelva el código de salida (0 = éxito, 1 = error). Los sistemas operativos usan esto para saber si un programa funcionó bien.

---

# LECCIÓN 6: EL SDK (llm_sdk)

## 6.1. ¿Qué es el SDK?

SDK significa "Software Development Kit" (kit de desarrollo de software). Es una caja de herramientas que nos da acceso al modelo sin tener que escribir todo el código complejo de bajo nivel.

## 6.2. La clase Small_LLM_Model

### __init__(self, model_name, device, dtype)

Carga el modelo. Por defecto usa "Qwen/Qwen3-0.6B" de Hugging Face.

Cosas que hace:
1. Elige el dispositivo: GPU (si hay), o CPU
2. Elige la precisión: float16 en GPU, float32 en CPU
3. Carga el tokenizador (convierte texto ↔ tokens)
4. Carga el modelo
5. Pone el modelo en modo evaluación (no entrenamiento)

### encode(text)

Convierte texto a IDs de tokens.

### decode(ids)

Convierte IDs de tokens a texto.

### get_logits_from_input_ids(input_ids)

**El método más importante para nosotros**. Recibe una lista de IDs y devuelve los logits para el siguiente token.

### get_path_to_vocab_file()

Descarga y devuelve la ruta al archivo vocab.json del modelo.

---

# LECCIÓN 7: EJEMPLOS PRÁCTICOS (NO SON DEL PROYECTO)

## Ejemplo 1: Tokenizador simple

Vamos a crear un tokenizador muy básico que divide el texto en palabras y espacios:

```python
def tokenizar_simple(texto: str) -> list[str]:
    """Divide el texto en palabras y espacios (muy básico)."""
    tokens = []
    palabra_actual = ""
    
    for char in texto:
        if char == " ":
            if palabra_actual:
                tokens.append(palabra_actual)
                palabra_actual = ""
            tokens.append(" ")  # El espacio es un token
        else:
            palabra_actual += char
    
    if palabra_actual:
        tokens.append(palabra_actual)
    
    return tokens

# Prueba
print(tokenizar_simple("Hola mundo"))
# ['Hola', ' ', 'mundo']
```

Este tokenizador es muy simple, pero ilustra la idea básica: el texto se divide en trozos (tokens) que el modelo puede entender.

## Ejemplo 2: Diccionario token ↔ ID

Vamos a crear un diccionario que asocia tokens con IDs numéricos:

```python
class VocabularioSimple:
    def __init__(self):
        self.token_a_id = {}
        self.id_a_token = {}
    
    def agregar(self, token: str):
        if token not in self.token_a_id:
            nuevo_id = len(self.token_a_id)
            self.token_a_id[token] = nuevo_id
            self.id_a_token[nuevo_id] = token
    
    def obtener_id(self, token: str) -> int | None:
        return self.token_a_id.get(token)
    
    def obtener_token(self, id: int) -> str | None:
        return self.id_a_token.get(id)

# Prueba
vocab = VocabularioSimple()
for token in ["Hola", "mundo", "¿", "?"]:
    vocab.agregar(token)

print(vocab.obtener_id("mundo"))   # 1
print(vocab.obtener_token(2))      # "¿"
```

Este diccionario es como un traductor entre texto y números. El modelo solo entiende números, así que necesitamos esta traducción.

## Ejemplo 3: Softmax y argmax

Vamos a implementar la función softmax y argmax desde cero:

```python
import math

def softmax(logits: list[float]) -> list[float]:
    """Convierte puntuaciones en probabilidades."""
    max_logit = max(logits)
    exp = [math.exp(x - max_logit) for x in logits]
    suma = sum(exp)
    return [x / suma for x in exp]

def argmax(lista: list[float]) -> int:
    """Devuelve el índice del valor máximo."""
    return lista.index(max(lista))

# Prueba
logits = [2.0, 1.0, 0.5]
probs = softmax(logits)
print(probs)  # [0.659, 0.242, 0.099]
print(argmax(probs))  # 0 (el primer token es el más probable)
```

La softmax convierte puntuaciones en bruto en probabilidades que suman 1. El argmax elige el token con la probabilidad más alta.

## Ejemplo 4: Máscara simple

Vamos a implementar la máscara que se usa en la decodificación restringida:

```python
import math

def aplicar_mascara(logits: list[float], ids_validos: list[int]) -> list[float]:
    """Pone -infinito a los tokens que no están en ids_validos."""
    mascara = [-math.inf] * len(logits)
    for id_valido in ids_validos:
        if 0 <= id_valido < len(logits):
            mascara[id_valido] = logits[id_valido]
    return mascara

# Prueba
logits = [5.2, 3.1, 0.5, 0.1]
ids_validos = [0, 2]  # Solo permiten el token 0 y el 2
mascara = aplicar_mascara(logits, ids_validos)
print(mascara)  # [5.2, -inf, 0.5, -inf]
```

La máscara pone -infinito a los tokens que no queremos permitir. Así, el modelo no puede elegirlos nunca.

## Ejemplo 5: Decodificación restringida paso a paso

Vamos a simular cómo funciona la decodificación restringida en un caso simple:

```python
# Supongamos que el modelo tiene 5 tokens en su vocabulario
# y queremos generar un número de 2 dígitos

vocabulario = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-", "\n"]

# El modelo devuelve logits para cada token
logits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.3, 0.2, 0.5]

# Solo queremos permitir dígitos (IDs 0-9) y el salto de línea (ID 12)
ids_validos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]

# Aplicamos la máscara
mascara = aplicar_mascara(logits, ids_validos)
print(mascara)
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -inf, -inf, 0.5]

# El modelo elige el token con la puntuación más alta
mejor_id = argmax(mascara)
print(mejor_id)  # 9 (el token "9")

# Repetimos el proceso para el siguiente token...
```

Este ejemplo muestra cómo la decodificación restringida funciona en la práctica: en cada paso, solo se permiten ciertos tokens, y el modelo elige el mejor de esos.

---

# LECCIÓN 8: PREGUNTAS FRECUENTES

## ¿Por qué no puedo usar transformers o pytorch?

Porque el proyecto quiere que aprendas a implementar la decodificación restringida desde cero, no que uses una librería que ya lo hace por ti.

## ¿Por qué no puedo usar heurísticas para elegir la función?

Porque el objetivo es que el **modelo** elija la función basándose en su comprensión del lenguaje, no tú con reglas manuales como "si la pregunta contiene 'suma', elige fn_sumar".

## ¿Qué es llm_sdk?

Es un **wrapper** (envoltorio) que te dan para interactuar con el modelo Qwen3-0.6B sin tener que instalar pytorch ni transformers. Es como un "control remoto" para el modelo.

## ¿Qué es uv?

Es un gestor de paquetes de Python más rápido que pip. Se usa así:
- uv sync: Instala dependencias
- uv run python: Ejecuta Python con el entorno virtual

## ¿Qué es Pydantic?

Es una librería que valida datos automáticamente. Si dices que un campo debe ser un string y le pasas un número, Pydantic lanza un error claro.

## ¿Por qué no usar simplemente json.dumps()?

Porque el objetivo del proyecto es que sea el MODELO quien elija los valores. Si usáramos json.dumps() directamente, estaríamos poniendo valores fijos, no valores elegidos por la IA.

## ¿Por qué no dejar al modelo escribir JSON libremente?

Porque el modelo Qwen3-0.6B es pequeño y falla el 70% de las veces. El proyecto evalúa específicamente la capacidad de implementar **decodificación restringida**, no la capacidad de hacer prompting.

## ¿Qué pasa si el modelo no encuentra el valor correcto?

El sistema siempre genera algo porque siempre hay tokens válidos disponibles. Pero el valor puede ser incorrecto. Por ejemplo, para "Greet shrek", el modelo podría extraer "shrek" (correcto) o "Shrek" (con mayúscula, igualmente válido pero diferente).

El proyecto no exige que los valores sean perfectos, sino que el sistema genere JSON **válido** mediante decodificación restringida.

## ¿Por qué dos fases (selección + extracción)?

1. **Separación de responsabilidades**: cada fase hace una cosa simple
2. **Menos errores**: si el modelo genera todo de golpe, un error en el nombre de la función arruina todo. Separando, si la extracción de un argumento falla, los demás siguen funcionando
3. **Prompts más simples**: cada prompt pide una cosa concreta, lo que facilita que el modelo acierte

## ¿Qué es la "máscara de logits"?

Es la técnica de poner -infinito en los logits de los tokens que no queremos permitir. Es como poner una máscara sobre las opciones prohibidas para que el modelo no pueda verlas.

## ¿Por qué -infinito y no 0?

Porque el modelo elige el token con el logit MÁS ALTO. Si pusiéramos 0, algunos tokens prohibidos podrían tener logits negativos (por ejemplo, -5) y el 0 sería mayor que esos, con lo que el token prohibido sería elegido. Con -infinito, ningún token prohibido puede ser elegido jamás.

## ¿Qué significa model.encode().flatten().tolist()?

1. model.encode(prompt) → convierte texto a tensor 2D: [[1, 2, 3, 4]]
2. .flatten() → aplana a 1D: [1, 2, 3, 4]
3. .tolist() → convierte a lista Python: [1, 2, 3, 4]

Necesitamos una lista plana para pasársela a get_logits_from_input_ids.

---

# LECCIÓN 9: GLOSARIO

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

# LECCIÓN 10: RESUMEN FINAL

## Lo que has aprendido

1. **Qué es el function calling**: convertir lenguaje humano en llamadas a funciones estructuradas
2. **Por qué los modelos pequeños son torpes**: fallan el 70% de las veces cuando se les pide JSON
3. **Qué es la decodificación restringida**: filtrar tokens inválidos para que el modelo no pueda equivocarse
4. **Cómo funciona un modelo de lenguaje**: tokens, IDs, logits, softmax, argmax
5. **Cómo se organiza el proyecto**: models.py, vocab.py, constrained_decoder.py, function_caller.py, prompt_builder.py, tools.py, __main__.py
6. **Qué es el SDK**: una caja de herramientas para usar el modelo sin instalar pytorch
7. **Por qué usamos Pydantic**: para validar que los datos tengan el formato correcto

## Lo que vas a construir

Un sistema que:
1. Lee funciones disponibles y preguntas de prueba
2. Usa un modelo de lenguaje para elegir la función correcta
3. Extrae los argumentos de cada función usando decodificación restringida
4. Genera un archivo JSON con los resultados

## El objetivo

No es terminar rápido. Es **entender** cada paso. Si entiendes la decodificación restringida, puedes aplicarla a cualquier problema donde necesites que un modelo genere texto con un formato específico.

---

# APÉNDICE: EJERCICIOS PARA PRACTICAR

## Ejercicio 1: Tokenizador

Crea un tokenizador que divida el texto en palabras, signos de puntuación y espacios. Prueba con diferentes textos y observa cómo cambian los tokens.

## Ejercicio 2: Diccionario

Crea un diccionario que asocie tokens con IDs. Añade 10 tokens y prueba a buscar sus IDs.

## Ejercicio 3: Softmax

Implementa la función softmax y pruébala con diferentes listas de logits. Observa cómo cambian las probabilidades.

## Ejercicio 4: Máscara

Implementa la función de máscara y pruébala con diferentes listas de tokens válidos. Observa cómo cambian los logits.

## Ejercicio 5: Decodificación restringida

Simula un caso simple de decodificación restringida: genera un número de 2 dígitos usando solo tokens de dígitos.

---

**Fin de la guía. ¡Mucha suerte con el proyecto!**
