# GUÍA COMPLETA: Call Me Maybe

## El proyecto explicado como si nunca hubieras programado en tu vida

---

# PARTE 1: ¿QUÉ ESTAMOS CONSTRUYENDO?

## El problema del mundo real

Imagina que tienes una calculadora científica con 5 botones especiales. Cada botón hace una cosa:

- Botón A: suma dos números
- Botón B: saluda a una persona por su nombre
- Botón C: invierte una palabra (la pone al revés)
- Botón D: calcula la raíz cuadrada de un número
- Botón E: busca y reemplaza texto usando patrones

La calculadora no entiende español ni inglés. Solo entiende órdenes muy estructuradas, como:

```
BOTÓN: fn_add_numbers
VALOR1: 40
VALOR2: 2
```

Ahora imagina que tienes un amigo que habla lenguaje normal. Te dice:

> "Oye, ¿cuánto es 40 más 2?"

Tú necesitas traducir eso a:

```
BOTÓN: fn_add_numbers
VALOR1: 40
VALOR2: 2
```

Eso es EXACTAMENTE lo que hace este proyecto. Toma lenguaje humano y lo convierte en la llamada correcta a una función, con los valores correctos para cada botón.

## El formato exacto de la respuesta

El programa debe escribir un archivo JSON (un formato de datos muy usado) que se ve así:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

Fíjate bien en las 3 partes:
- **prompt**: la pregunta original que hizo el usuario
- **fn_name**: qué botón (función) hay que pulsar
- **args**: los valores concretos que necesita esa función

## ¿Por qué no podemos simplemente preguntarle a la inteligencia artificial?

Aquí viene el problema gordo. El modelo que usamos es **Qwen3-0.6B**. Tiene 600 millones de parámetros, que suena a mogollón. Pero para que te hagas una idea, el ChatGPT más moderno tiene más de 1 billón (1.000.000.000.000). Nuestro modelo es diminuto en comparación.

Si le dices al modelo:

> "Escribe un JSON válido con la función y los argumentos"

fallará el 70% de las veces. Escribirá cosas como:

```json
{"function": "add_numbers"}   ← le falta "fn_name" 
```

o

```json
{fn_name: add}                ← le faltan las comillas
```

o directamente escribirá:

```
The result of adding 2 and 3 is 5
```

Que no es lo que queremos.

## La solución: no dejar que el modelo se equivoque

La idea genial del proyecto es: **no le preguntes al modelo qué quiere escribir. Oblígalo a escribir solo lo que es válido.**

Cada vez que el modelo va a escribir una palabra (técnicamente un "token"), el programa le dice:

> "De las 150.000 palabras que te sé, solo te dejo elegir entre estas 3 que son válidas. Elige la que más te guste de esas 3."

Así es **físicamente imposible** que el modelo escriba algo incorrecto. Es como si a un niño le das a elegir entre "manzana", "pera" y "plátano" — puede elegir cualquiera, pero nunca va a elegir "coche" porque no está entre las opciones.

Eso se llama **decodificación restringida** (constrained decoding en inglés) y es el corazón del proyecto.

---

# PARTE 2: ¿CÓMO PIENSA UNA INTELIGENCIA ARTIFICIAL?

Antes de entender el código, necesitas entender cómo funciona un modelo de lenguaje por dentro.

## Tokens: los ladrillos del lenguaje

Los modelos NO entienden letras. Tampoco entienden palabras exactamente. Entienden **tokens**.

Un token es un trozo de texto. Puede ser:

- Una palabra entera: `hello`, `world`, `42`
- Parte de una palabra: `un`, `believable` (para "unbelievable")
- Un signo: `{`, `:`, `.`
- Un espacio: ` hola` (sí, el espacio cuenta como parte del token)

El modelo Qwen3-0.6B tiene un vocabulario de aproximadamente **150.000 tokens diferentes**. Cada token tiene un número de identificación (ID).

Por ejemplo:
```
Token "hello"   → ID 334
Token " world"  → ID 335
Token "{"       → ID 42
Token "}"       → ID 43
Token "fn_add"  → ID 15782
```

## Input IDs: el modelo solo entiende números

Cuando le pasas texto al modelo, primero lo conviertes a números (IDs de tokens):

```
Texto: "What is 2+3?"
       ↓
Tokenización: ["What", " is", " 2", "+", "3", "?"]
       ↓
IDs: [1284, 339, 456, 27, 457, 30]
```

Esa lista de números es lo que el modelo recibe. Se llama **input_ids**.

## Logits: las puntuaciones del modelo

Cuando el modelo recibe los IDs, no te devuelve directamente "la siguiente palabra es X". Te devuelve una **puntuación** para cada uno de los 150.000 tokens de su vocabulario.

Esa puntuación se llama **logit** (o logit). Es un número que puede ser positivo o negativo. Cuanto más alto, más probable cree el modelo que es ese token.

Imagínate que el modelo recibe:

```
Input IDs: [1284, 339, 456, 27, 457, 30]   ← "What is 2+3?"
```

Y devuelve algo así (simplificado a 5 tokens en vez de 150.000):

```
Token "the"  → logit = 0.3
Token " is"  → logit = 2.1
Token "?"    → logit = 5.4   ← el más alto
Token "sum"  → logit = 1.8
Token "{"    → logit = 0.1
```

El modelo cree que lo más probable después de "What is 2+3?" es "?". Tiene sentido: es una pregunta.

Normalmente, la IA elegiría el token con el logit más alto (el 5.4, que es "?"). Pero nosotros vamos a **modificar** esas puntuaciones.

## Softmax: convertir puntuaciones en probabilidades

Los logits son números en bruto, difíciles de interpretar. Para convertirlos en probabilidades (números entre 0 y 1 que suman 1), se usa **softmax**.

La fórmula es: $softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

No te asustes con la fórmula. Lo que hace es:
1. Coge cada logit
2. Calcula el número $e^{logit}$ (e ≈ 2.71828 elevado al logit)
3. Divide cada uno por la suma de todos

El resultado: probabilidades que suman 1. Un token con probabilidad 0.8 significa que el modelo cree que hay un 80% de probabilidades de que sea ese.

En el proyecto usamos **log-softmax**, que es el logaritmo de la softmax. Es más estable numéricamente y más práctico para sumar probabilidades (en vez de multiplicarlas).

---

# PARTE 3: ARQUITECTURA DEL PROYECTO

## El mapa de los archivos

```
call-me-maybe/
├── src/                         ← Todo el código fuente está aquí
│   ├── __init__.py              ← Marca la carpeta como un módulo Python
│   ├── __main__.py              ← Punto de entrada (el que se ejecuta con "uv run python -m src")
│   ├── models.py                ← Las "plantillas" de datos (Pydantic)
│   ├── vocab.py                 ← El diccionario token ↔ ID
│   ├── constrained_decoder.py   ← El CORAZÓN del proyecto (decodificación restringida)
│   ├── function_caller.py       ← El ORQUESTADOR (coordina todo)
│   ├── prompt_builder.py        ← Construye los mensajes para la IA
│   └── tools.py                 ← Lee y escribe archivos
├── llm_sdk/                     ← El SDK (caja de herramientas) para usar el modelo
│   └── llm_sdk/
│       └── __init__.py          ← Contiene la clase Small_LLM_Model
├── data/
│   ├── input/
│   │   ├── function_definitions.json   ← Las funciones disponibles (botones)
│   │   └── function_calling_tests.json ← Las preguntas de prueba
│   └── output/
│       └── function_calling_results.json ← Las respuestas generadas
├── Makefile                     ← Atajos para comandos frecuentes
├── pyproject.toml               ← Configuración del proyecto Python
└── README.md                    ← Este archivo
```

## El flujo de principio a fin

Vamos a seguir el viaje de una pregunta desde que entra hasta que sale:

```
Pregunta: "What is the sum of 2 and 3?"
                      ↓
          1. LEER ARCHIVOS DE ENTRADA
             (tools.py: json_reader, load_function_def, load_prompt)
                      ↓
          2. CARGAR EL MODELO
             (llm_sdk: Small_LLM_Model → Qwen3-0.6B)
                      ↓
          3. CONSTRUIR PROMPT DE SELECCIÓN
             (prompt_builder.py: build_function_selection_prompt)
             → "You are a function calling assistant...
                Available functions: fn_add_numbers, fn_greet...
                Function to call: "
                      ↓
          4. ELEGIR FUNCIÓN
             (constrained_decoder.py: JSONGenerator.select_function_name)
             → Puntúa cada nombre de función con log-probabilidades
             → Elige "fn_add_numbers" (tiene la puntuación más alta)
                      ↓
          5. PARA CADA PARÁMETRO, EXTRAER VALOR
             (function_caller.py: bucle sobre parámetros)
                      ↓
          5a. Parámetro "a":
              - Construir prompt de extracción
                → "Extract parameter 'a' (type: number): Value: "
              - Generar número con decodificación restringida
                → Solo permite dígitos, punto decimal y signo negativo
                → El modelo genera "4", "0", luego detecta que termina
                → Resultado: 40.0
                      ↓
          5b. Parámetro "b":
              - Mismo proceso
                → Resultado: 2.0
                      ↓
          6. ENSAMBLAR RESULTADO
             → {"prompt": "What is the sum of 2 and 3?",
                "fn_name": "fn_add_numbers",
                "args": {"a": 40.0, "b": 2.0}}
                      ↓
          7. GUARDAR EN ARCHIVO
             (tools.py: json_exporter)
             → data/output/function_calling_results.json
```

---

# PARTE 4: EXPLICACIÓN DETALLADA DE CADA ARCHIVO

---

## 4.1. `src/models.py` — Las plantillas de datos

### ¿Qué hace?

Define las **formas** que tienen que tener los datos. Como esos moldes de galletas: si el dato no encaja en el molde, el programa se da cuenta y lanza un error.

### ¿Por qué Pydantic?

Pydantic es una librería que valida datos automáticamente. Imagina que el archivo de funciones tiene:

```json
{
  "name": "fn_add_numbers",
  "type": 42
}
```

Cuando Pydantic intente meter eso en la plantilla, verá que "type" debería ser un texto ("number", "string"...), no un número 42. Y lanzará un error clarísimo: "el campo 'type' debe ser de tipo str, no int".

Sin Pydantic, ese error pasaría desapercibido y el programa fallaría más tarde de forma misteriosa.

### Las 3 plantillas

**ParameterDefinition** — Define UN parámetro de una función:

```python
class ParameterDefinition(BaseModel):
    type: str
```

Cada parámetro solo tiene un campo: `type`, que puede ser "number", "string", etc. Es como decir: "el parámetro 'a' es de tipo número".

**FunctionDefinition** — Define UNA función completa:

```python
class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, Any]
```

Tiene:
- `name`: el nombre (ej: "fn_add_numbers")
- `description`: qué hace (ej: "Add two numbers...")
- `parameters`: un diccionario de parámetros. Las claves son los nombres ("a", "b") y los valores son objetos ParameterDefinition
- `returns`: qué devuelve la función

El `@field_validator("parameters")` es un "validador personalizado". Se ejecuta ANTES de guardar los datos. Su trabajo es convertir los diccionarios JSON en objetos ParameterDefinition automáticamente.

**FunctionCall** — El resultado final:

```python
class FunctionCall(BaseModel):
    prompt: str
    fn_name: str
    args: dict[str, Any]
```

Representa exactamente lo que queremos generar:
- `prompt`: la pregunta original
- `fn_name`: el nombre de la función elegida
- `args`: los argumentos extraídos

---

## 4.2. `src/vocab.py` — El diccionario token ↔ ID

### ¿Qué hace?

Carga el archivo `vocab.json` del modelo (que puede tener 150.000 entradas) y construye un índice de búsqueda en ambas direcciones:

1. **ID → Token**: dado un ID numérico, ¿qué texto representa?
   - `id_to_token[334]` → `"hello"`
2. **Token → IDs**: dado un texto, ¿qué IDs lo representan?
   - `token_to_ids["hello"]` → `[334, 335]` (puede haber múltiples IDs para el mismo texto)

### El problema del formato

El archivo `vocab.json` puede venir en dos formatos distintos:

**Formato A** (clave = token, valor = ID):
```json
{"hello": 334, "world": 335, "{": 42}
```

**Formato B** (clave = ID, valor = token):
```json
{"334": "hello", "335": "world", "42": "{"}
```

El código detecta automáticamente cuál es. Mira la primera clave del archivo. Si parece un número (como "334"), usa el formato B. Si parece texto (como "hello"), usa el formato A.

```python
first_key = next(iter(raw.keys()))
try:
    int(first_key)  # ¿La primera clave es un número?
    # Formato B: clave = ID, valor = token
    self.id_to_token = {int(k): v for k, v in raw.items()}
except (ValueError, TypeError):
    # Formato A: clave = token, valor = ID
    self.id_to_token = {int(v): k for k, v in raw.items()}
```

### Métodos de búsqueda

- `get_ids_exact(":")` → busca tokens que sean exactamente ":"
- `get_ids_chars("0123456789")` → busca tokens compuestos solo por esos caracteres (dígitos)
- `get_ids_start_with("fn_")` → busca tokens que empiecen por "fn_"

---

## 4.3. `src/constrained_decoder.py` — EL CORAZÓN DEL PROYECTO

### ¿Qué hace?

Este archivo implementa la **decodificación restringida**. Es la parte más importante y la que realmente evalúa el proyecto.

### Las funciones auxiliares

**`get_next_token_logits(model, input_ids)`**

Le pide al modelo las puntuaciones (logits) para el siguiente token.

El modelo recibe la lista de IDs y devuelve 150.000 números (uno por cada token de su vocabulario). Convertimos esos números a un array de NumPy para poder manipularlos fácilmente.

**`apply_mask(logits, valid_ids)`**

Esta función ES la decodificación restringida en su forma más pura:

```python
def apply_mask(logits, valid_ids):
    masked = np.full(len(logits), -math.inf)
    for tid in valid_ids:
        if 0 <= tid < len(logits):
            masked[tid] = logits[tid]
    return masked
```

Lo que hace:
1. Crea un array nuevo lleno de `-infinito` (menos infinito)
2. Para cada ID que SÍ es válido, copia su logit original a esa posición
3. Los tokens inválidos se quedan con `-infinito`

Resultado: los tokens inválidos **nunca** serán elegidos, porque `-infinito` es la puntuación más baja posible.

**`select_best_token(masked_logits)`**

Elige el token con la puntuación más alta del array enmascarado. Si todos son `-infinito` (caso de error), devuelve `None`.

### La clase JSONGenerator

Esta clase contiene la lógica de generación con restricciones. Tiene los siguientes métodos:

#### `select_function_name(prompt_ids, candidate_names)`

**Escenario**: Tenemos una pregunta como "What is the sum of 2 and 3?" y 5 funciones disponibles.

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

Todos empiezan igual, así que necesitamos ver los tokens siguientes. El método `_score_sequence` hace forward passes adicionales para tokens posteriores del nombre.

```python
def _score_sequence(self, context_ids, target_ids, first_logprobs):
    # Primer token: usar los log-probs ya calculados
    score = first_logprobs[target_ids[0]]
    
    # Tokens siguientes: pedir nuevos logits al modelo
    for next_id in target_ids[1:]:
        nuevos_logits = get_next_token_logits(model, context + [token_anterior])
        score += log_softmax(nuevos_logits)[next_id]
    
    return score
```

Al final, si la pregunta es sobre sumar números, `fn_add_numbers` tendrá la puntuación más alta.

#### `_log_softmax(logits)`

Convierte logits en bruto a log-probabilidades. La implementación es **numéricamente estable**: resta el máximo antes de hacer exponenciales para evitar que los números se disparen.

```python
def _log_softmax(self, logits):
    max_logit = np.max(logits)    # Encontrar el máximo
    shifted = logits - max_logit   # Restarlo (los números se centran en 0)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))  # Log de la suma de exponenciales
    return shifted - log_sum_exp   # Resultado: log-probabilidades
```

#### `extract_number(prompt_ids)`

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

```
Paso 1: prompt = "Extract parameter 'a' (type: number):\nValue: "
        Tokens válidos: ["0", "1", ..., "9", "-"]
        El modelo elige: "4"
        Acumulado: "4"

Paso 2: Tokens válidos: ["0", "1", ..., "9", "."]
        El modelo elige: "0"
        Acumulado: "40"

Paso 3: Tokens válidos: ["0", "1", ..., "9", "."]
        El modelo elige: "."
        Acumulado: "40."

Paso 4: Tokens válidos: ["0", "1", ..., "9"]
        El modelo elige: "0"
        Acumulado: "40.0"

Paso 5: Tokens válidos: ["0", "1", ..., "9", ".", terminadores]
        El modelo elige: "," (terminador)
        → Se para, devuelve float("40.0") = 40.0
```

#### `extract_string(prompt_ids)`

**Escenario**: Tenemos que extraer "shrek" para el parámetro 'name'.

**Proceso**:
1. Creamos una lista de tokens válidos: TODOS excepto los que rompen la estructura JSON (`{`, `}`, `[`, `]`)
2. Definimos tokens que indican el fin del string (`\n`, `",`, `"}`)
3. En cada paso:
   - Aplicamos la máscara
   - El modelo elige un token
   - Si termina en fin-de-string: paramos
   - Si no: acumulamos

**Limitación**: Este método es el más simple. Para valores string que contienen comillas o caracteres especiales, puede fallar. Pero para los casos de prueba del proyecto (nombres, palabras simples) funciona bien.

#### `extract_boolean(prompt_ids)`

**Escenario**: Tenemos que decidir entre "true" o "false".

**Proceso**:
1. Obtenemos los logits y calculamos log-probabilidades
2. Codificamos "true" y "false" a IDs
3. Puntuamos la secuencia completa de "true" y "false"
4. Devolvemos True si "true" tiene mayor puntuación

---

## 4.4. `src/prompt_builder.py` — Construir los mensajes para la IA

### ¿Qué hace?

Construye los textos (prompts) que le pasamos al modelo. La forma en que le preguntamos al modelo afecta MUCHO a la respuesta.

### `build_function_selection_prompt`

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

### `build_argument_extraction_prompt`

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

---

## 4.5. `src/function_caller.py` — El orquestador

### ¿Qué hace?

Coordina todo el proceso. Es como el director de orquesta: no toca ningún instrumento, pero sabe cuándo debe entrar cada uno.

```python
class FunctionCaller:
    def __init__(self, model):
        self.model = model
        self.vocab = VocabIndex(model)          # Carga el vocabulario
        self.generator = JSONGenerator(model, self.vocab)  # Crea el generador
    
    def resolve(self, prompt, functions):
        # FASE 1: Elegir función
        selection_prompt = build_function_selection_prompt(prompt, functions)
        selection_ids = model.encode(selection_prompt)
        chosen_name = self.generator.select_function_name(selection_ids, nombres)
        
        # FASE 2: Extraer cada argumento
        for cada parámetro:
            arg_prompt = build_argument_extraction_prompt(prompt, fn_def, param, args_ya_extraidos)
            arg_ids = model.encode(arg_prompt)
            
            if type == "number":
                valor = self.generator.extract_number(arg_ids)
            elif type == "boolean":
                valor = self.generator.extract_boolean(arg_ids)
            else:
                valor = self.generator.extract_string(arg_ids)
        
        # FASE 3: Ensamblar
        return FunctionCall(prompt=prompt, fn_name=chosen_name, args=args)
```

### ¿Por qué dos fases?

Podríamos intentar que el modelo generara todo el JSON de golpe. Pero eso es mucho más complejo (habría que trackear el estado del JSON token a token) y más propenso a errores con un modelo pequeño.

Dividir en fases:
1. La función se elige por puntuación (solo entre nombres válidos)
2. Cada argumento se extrae por separado con restricciones específicas según su tipo

Esto hace que cada paso sea simple y que el modelo solo tenga que hacer una cosa a la vez.

---

## 4.6. `src/tools.py` — Leer y escribir archivos

### ¿Qué hace?

Operaciones de entrada/salida: leer los archivos JSON y escribir los resultados.

### `json_reader(path)`

Lee un archivo JSON y lo convierte en datos Python. Si el archivo no existe, lanza error. Si el JSON está mal formado, también.

```python
def json_reader(path):
    if not path.exists():
        raise FileNotFoundError("Archivo no encontrado")
    
    with open(path, 'r') as f:
        return json.load(f)  # Si el JSON está roto, falla aquí
```

### `load_function_def(path)`

Carga las definiciones de funciones y las valida con Pydantic.

Lee el JSON, verifica que sea una lista, y para cada elemento crea un objeto `FunctionDefinition`. Si alguna definición es inválida, indica en qué índice falló.

### `load_prompt(path)`

Carga los prompts de prueba. El formato esperado es:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  ...
]
```

También acepta strings directamente: `["What is...", "Greet..."]`

### `json_exporter(results, path)`

Escribe los resultados en un archivo JSON. Crea la carpeta si no existe.

```python
def json_exporter(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = [result.model_dump() for result in results]
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
```

El `indent=2` hace que el JSON se vea bonito con sangría de 2 espacios.

---

## 4.7. `src/__main__.py` — El punto de entrada

### ¿Qué hace?

Es el archivo que se ejecuta cuando escribes `uv run python -m src` (el `-m src` busca `src/__main__.py`).

### `parse_args()`

Define los argumentos que acepta el programa:

```
--input   Directorio de entrada (por defecto: data/input)
--output  Archivo de salida (por defecto: data/output/function_calling_results.json)
```

### `main()`

El flujo completo:

1. Parsear argumentos de línea de comandos
2. Determinar las rutas de los archivos de entrada:
   - `function_definitions.json` (las funciones)
   - `function_calling_tests.json` (los prompts)
3. Cargar archivos (si falla, salir con código 1)
4. Cargar el modelo Qwen3-0.6B
5. Crear el `FunctionCaller`
6. Para cada prompt:
   - Extraer el texto del prompt (puede venir como string o como diccionario)
   - Llamar a `caller.resolve()`
   - Si falla: contar error y continuar
7. Escribir los resultados
8. Mostrar resumen
9. Devolver 0 si todo fue bien, 1 si hubo errores

### El bloque `if __name__ == "__main__"`

```python
if __name__ == "__main__":
    sys.exit(main())
```

Este bloque es estándar en Python. Significa: "Si este archivo se ejecuta directamente (no importado), ejecuta `main()`".

El `sys.exit()` hace que el programa devuelva el código de salida (0 = éxito, 1 = error). Los sistemas operativos usan esto para saber si un programa funcionó bien.

---

# PARTE 5: EL SDK (llm_sdk)

## ¿Qué es el SDK?

SDK significa "Software Development Kit" (kit de desarrollo de software). Es una caja de herramientas que nos da acceso al modelo sin tener que escribir todo el código complejo de bajo nivel.

## La clase Small_LLM_Model

### `__init__(self, model_name, device, dtype)`

Carga el modelo. Por defecto usa "Qwen/Qwen3-0.6B" de Hugging Face.

Cosas que hace:
1. Elige el dispositivo: GPU (si hay), o CPU
2. Elige la precisión: float16 en GPU, float32 en CPU
3. Carga el tokenizador (convierte texto ↔ tokens)
4. Carga el modelo
5. Pone el modelo en modo evaluación (no entrenamiento)

### `encode(text)`

Convierte texto a IDs de tokens.

```python
encode("hola") → [[1284, 339]]  # Tensor 2D
```

### `decode(ids)`

Convierte IDs de tokens a texto.

```python
decode([1284, 339]) → "hola"
```

### `get_logits_from_input_ids(input_ids)`

**El método más importante para nosotros**. Recibe una lista de IDs y devuelve los logits para el siguiente token.

```python
get_logits_from_input_ids([1284, 339]) → [0.3, 2.1, 5.4, ...]  # 150.000 números
```

### `get_path_to_vocab_file()`

Descarga y devuelve la ruta al archivo `vocab.json` del modelo.

---

# PARTE 6: LOS ARCHIVOS DE DATOS

## `data/input/function_definitions.json`

Define las 5 funciones "botones" que el sistema puede usar. Cada función tiene:
- `name`: el nombre (ej: "fn_add_numbers")
- `description`: qué hace (ej: "Add two numbers together and return their sum.")
- `parameters`: los parámetros que necesita. Cada uno con su tipo ("number" o "string")
- `returns`: qué tipo de valor devuelve

## `data/input/function_calling_tests.json`

Las 11 preguntas de prueba que el sistema debe procesar. Van desde simples sumas hasta reemplazos con expresiones regulares.

## `data/output/function_calling_results.json`

El archivo de salida con los resultados generados. Se crea automáticamente.

---

# PARTE 7: EL MAKEFILE

## ¿Qué es un Makefile?

Un Makefile es un archivo con "ataques" (atajos) para comandos frecuentes. En vez de escribir `PYTHONPATH=. uv run python3 -m src --input data/input --output data/output/results.json`, solo escribes `make run`.

### Comandos disponibles

```bash
make        # Equivale a make install + make run
make install  # Instala dependencias
make run      # Ejecuta el programa
make lint     # Comprueba calidad del código (flake8 + mypy)
make clean    # Limpia archivos temporales
```

---

# PARTE 8: PREGUNTAS FRECUENTES

## ¿Por qué no usar simplemente `json.dumps()`?

Porque el objetivo del proyecto es que sea el MODELO quien elija los valores. Si usáramos `json.dumps()` directamente, estaríamos poniendo valores fijos, no valores elegidos por la IA.

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

Es la técnica de poner `-infinito` en los logits de los tokens que no queremos permitir. Es como poner una máscara sobre las opciones prohibidas para que el modelo no pueda verlas.

## ¿Por qué `-infinito` y no `0`?

Porque el modelo elige el token con el logit MÁS ALTO. Si pusiéramos 0, algunos tokens prohibidos podrían tener logits negativos (por ejemplo, -5) y el 0 sería mayor que esos, con lo que el token prohibido sería elegido. Con `-infinito`, ningún token prohibido puede ser elegido jamás.

## ¿Qué significa `model.encode().flatten().tolist()`?

1. `model.encode(prompt)` → convierte texto a tensor 2D: `[[1, 2, 3, 4]]`
2. `.flatten()` → aplana a 1D: `[1, 2, 3, 4]`
3. `.tolist()` → convierte a lista Python: `[1, 2, 3, 4]`

Necesitamos una lista plana para pasársela a `get_logits_from_input_ids`.

---

# PARTE 9: GLOSARIO

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
