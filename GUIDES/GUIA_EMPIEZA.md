# GUÍA PARA EMPEZAR: Call Me Maybe

## Tu primera hora: Entendiendo qué vas a construir

---

## 1. ¿Qué es "Function Calling"?

Imagina que tienes un asistente virtual muy inteligente, pero que solo sabe hablar. Le dices "¿qué tiempo hace en París?" y te responde "En París está soleado, 25°C". 

Pero, ¿y si quieres que el asistente no te lo diga, sino que llame a una función real que consulte el tiempo? Ahí entra el **function calling**:

```
Tú: "¿Qué tiempo hace en París?"

Sin function calling:
  Asistente: "En París hace sol y 25 grados"

Con function calling:
  Asistente: {
    "funcion": "obtener_tiempo",
    "argumentos": {"ciudad": "París"}
  }
```

El asistente NO te responde. En su lugar, te da una **receta estructurada** para que tu programa ejecute la función correcta con los argumentos correctos.

### Ejemplo simple con Python

```python
# Sin function calling: el modelo genera texto libre
respuesta = modelo.generar("¿Cuánto es 2+2?")
print(respuesta)  # "La suma de 2 y 2 es 4"

# Con function calling: el modelo genera una estructura
respuesta = modelo.function_call("¿Cuánto es 2+2?", funciones_disponibles)
print(respuesta)
# {"fn_name": "sumar", "args": {"a": 2, "b": 2}}
```

Tu programa después ejecuta `sumar(2, 2)` y obtiene `4`.

---

## 2. El problema: los modelos pequeños son torpes

Vas a usar un modelo llamado **Qwen3-0.6B**. Tiene 600 millones de parámetros. Suena mucho, pero es diminuto comparado con GPT-4 (que tiene billones).

Si le pides a este modelo pequeño que genere JSON directamente:

```python
prompt = "Genera un JSON con la función para sumar 2 y 3"
respuesta = modelo.generar(prompt)
```

El 70% de las veces responderá cosas como:

```
"La función es sumar con a=2 y b=3"     ← No es JSON
{"function": "sumar"}                    ← Faltan argumentos
{fn_name: sumar, args: {a: 2, b: 3}}     ← JSON inválido (sin comillas)
```

**Necesitas una forma de obligar al modelo a producir JSON válido al 100%.**

---

## 3. La solución: Decodificación Restringida

### ¿Qué es?

La **decodificación restringida** (constrained decoding) es una técnica que filtra las opciones del modelo en cada paso para que solo pueda elegir tokens válidos.

### Analogía: El menú del restaurante

Imagina que vas a un restaurante con un menú de 150 platos. El camarero (el modelo) te pregunta qué quieres.

**Sin restricción:** El camarero puede recomendarte cualquier plato, incluso los que no están en el menú.

**Con restricción:** Tú le dices "solo puedes elegir entre paella, sushi o pizza". Ahora es imposible que te recomiende una hamburguesa.

En el proyecto:
- El "menú" son los 150.000 tokens del vocabulario del modelo
- La "restricción" es: "solo puedes elegir tokens que mantengan el JSON válido"

### ¿Cómo funciona paso a paso?

```
Paso 1: El modelo genera puntuaciones para todos los tokens
        Token "{" → puntuación 5.2
        Token "}" → puntuación 3.1
        Token "a" → puntuación 0.5
        Token "z" → puntuación 0.1
        ... (150.000 puntuaciones)

Paso 2: Tú pones a -infinito los tokens que NO quieres
        Token "{" → 5.2      ← Válido, se queda
        Token "}" → -infinito ← Inválido (aún no toca cerrar)
        Token "a" → -infinito ← Inválido
        Token "z" → -infinito ← Inválido

Paso 3: El modelo solo puede elegir entre los tokens con puntuación real
        → Elige "{" (el único válido)

Paso 4: Repites el proceso para el siguiente token
```

---

## 4. Conceptos básicos que necesitas saber

### 4.1. Tokens: las piezas del lenguaje

Los modelos NO entienden letras ni palabras. Entienden **tokens**.

Un token es un trozo de texto. Puede ser:
- Una palabra: `"hola"`, `"mundo"`
- Parte de palabra: `"in", "creíble"` (para "increíble")
- Un signo: `"{"`, `"}"`, `":"`
- Un espacio: `" hola"` (el espacio cuenta)

**Ejemplo con un tokenizador simple:**

```python
texto = "¿Cuánto es 2+3?"
tokens = tokenizar(texto)
# Resultado: ["¿", "Cu", "á", "nto", " es", " 2", "+", "3", "?"]
```

Cada token tiene un número ID único:

```python
token_a_id = {"¿": 1, "Cu": 2, "á": 3, "nto": 4, " es": 5, " 2": 6, "+": 7, "3": 8, "?": 9}
```

### 4.2. Input IDs: el modelo solo entiende números

```python
texto = "Hola mundo"
ids = [1284, 339]  # El modelo recibe esto, no el texto
```

### 4.3. Logits: las puntuaciones del modelo

Cuando el modelo recibe IDs, devuelve una **puntuación** para cada token posible del vocabulario:

```python
logits = modelo.obtener_logits([1284, 339])
# logits = [0.3, 2.1, 5.4, 0.1, 0.8, ...]  # 150.000 números
#            ↑     ↑     ↑
#          token1 token2 token3
```

El token con la puntuación más alta es el que el modelo "cree" que sigue.

### 4.4. Softmax: de puntuaciones a probabilidades

Los logits son números en bruto. Para convertirlos en probabilidades (0 a 1), se usa **softmax**:

```python
import math

def softmax(logits):
    exp = [math.exp(x) for x in logits]
    suma = sum(exp)
    return [x / suma for x in exp]

# Ejemplo simple
logits = [2.0, 1.0, 0.5]
probs = softmax(logits)
# probs = [0.659, 0.242, 0.099]  # Suman 1.0
```

En el proyecto usamos **log-softmax** (logaritmo de softmax) porque es más estable numéricamente.

### 4.5. Argmax: elegir el mejor

```python
logits = [0.3, 2.1, 5.4, 0.1]
mejor_indice = logits.index(max(logits))  # 2 (el token con puntuación 5.4)
```

---

## 5. Estructura del proyecto

### 5.1. Los archivos que necesitas crear

```
call-me-maybe/
├── src/                         ← Tu código fuente
│   ├── __init__.py              ← Marca la carpeta como módulo
│   ├── __main__.py              ← Punto de entrada (se ejecuta con "python -m src")
│   ├── models.py                ← Plantillas de datos (Pydantic)
│   ├── vocab.py                 ← Diccionario token ↔ ID
│   ├── constrained_decoder.py   ← El CORAZÓN: decodificación restringida
│   ├── function_caller.py       ← Coordina todo el proceso
│   ├── prompt_builder.py        ← Construye los mensajes para el modelo
│   └── tools.py                 ← Lee y escribe archivos
├── llm_sdk/                     ← El SDK (te lo dan, no lo creas)
│   └── llm_sdk/
│       └── __init__.py          ← Contiene la clase Small_LLM_Model
├── data/
│   ├── input/
│   │   ├── function_definitions.json   ← Las funciones disponibles
│   │   └── function_calling_tests.json ← Las preguntas de prueba
│   └── output/                        ← Aquí se guardan los resultados
├── Makefile                     ← Atajos para comandos
├── pyproject.toml               ← Configuración del proyecto
└── README.md                    ← Documentación
```

### 5.2. El flujo del programa

```
1. LEER archivos de entrada (tools.py)
   ↓
2. CARGAR el modelo Qwen3-0.6B (llm_sdk)
   ↓
3. Para cada pregunta:
   ↓
   3a. CONSTRUIR prompt para elegir función (prompt_builder.py)
       ↓
   3b. ELEGIR función usando decodificación restringida (constrained_decoder.py)
       ↓
   3c. Para cada parámetro de la función:
       - CONSTRUIR prompt para extraer el valor
       - EXTRAER valor usando decodificación restringida
       ↓
   3d. ENSAMBLAR resultado (function_caller.py)
   ↓
4. GUARDAR resultados en JSON (tools.py)
```

---

## 6. Restricciones importantes del subject

| Regla | Qué significa para ti |
|-------|----------------------|
| **Python 3.10+** | Usa una versión moderna de Python |
| **flake8 + mypy** | Tu código debe pasar estas herramientas de calidad |
| **Pydantic** | Todas tus clases de datos deben usar Pydantic |
| **numpy + json** | Solo puedes usar estas librerías (además de la stdlib) |
| **PROHIBIDO pytorch, transformers, etc.** | No puedes usar librerías de ML directamente |
| **Qwen/Qwen3-0.6B** | Debes usar este modelo específico |
| **Sin heurísticas** | La función debe elegirla el LLM, no tú con if/else |
| **Sin métodos privados de llm_sdk** | Solo usa los métodos públicos del SDK |
| **uv para dependencias** | Usa `uv` en vez de `pip` |
| **Makefile obligatorio** | Debes crear un Makefile con reglas específicas |

---

## 7. Tu plan de trabajo (4-5 horas)

### Hora 1: Preparar el entorno

1. **Crear la estructura de carpetas**
   ```bash
   mkdir -p src llm_sdk data/input data/output GUIDES
   ```

2. **Crear `pyproject.toml`** con las dependencias:
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

3. **Crear el Makefile** con las reglas obligatorias:
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

4. **Copiar `llm_sdk`** en la raíz del proyecto (te lo proporcionan)

5. **Crear los archivos de datos de entrada** en `data/input/`

### Hora 2: Crear los modelos de datos (models.py)

Define las **plantillas** de tus datos con Pydantic:

```python
from pydantic import BaseModel, Field

class ParameterDefinition(BaseModel):
    """Define un parámetro de una función."""
    type: str  # "number", "string", "boolean"

class FunctionDefinition(BaseModel):
    """Define una función completa."""
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str]

class FunctionCall(BaseModel):
    """El resultado final: qué función llamar y con qué argumentos."""
    prompt: str
    fn_name: str
    args: dict[str, object]
```

### Hora 3: Crear el vocabulario (vocab.py)

Necesitas un diccionario que te permita:
- Dado un token (texto), encontrar su ID
- Dado un ID, encontrar su token (texto)

```python
class VocabIndex:
    def __init__(self, vocab_path: str):
        # Cargar el JSON del vocabulario
        # Crear diccionarios: token_a_id y id_a_token
        pass
    
    def get_ids_exact(self, texto: str) -> list[int]:
        """Devuelve los IDs de tokens que son exactamente 'texto'."""
        pass
    
    def get_ids_start_with(self, prefijo: str) -> list[int]:
        """Devuelve los IDs de tokens que empiezan por 'prefijo'."""
        pass
```

### Hora 4: El decodificador restringido (constrained_decoder.py)

Este es el **corazón** del proyecto. Necesitas:

1. **Una función para obtener logits del modelo:**
   ```python
   def obtener_logits(modelo, input_ids: list[int]) -> list[float]:
       """Pide al modelo las puntuaciones para el siguiente token."""
       pass
   ```

2. **Una función para aplicar la máscara:**
   ```python
   def aplicar_mascara(logits: list[float], ids_validos: list[int]) -> list[float]:
       """Pone -infinito a los tokens que NO están en ids_validos."""
       pass
   ```

3. **Una función para elegir el mejor token:**
   ```python
   def elegir_mejor_token(masked_logits: list[float]) -> int | None:
       """Devuelve el ID del token con mayor puntuación."""
       pass
   ```

4. **Una clase que coordine todo:**
   ```python
   class DecodificadorRestringido:
       def __init__(self, modelo, vocabulario: VocabIndex):
           self.modelo = modelo
           self.vocabulario = vocabulario
       
       def seleccionar_nombre_funcion(self, prompt_ids: list[int], nombres: list[str]) -> str:
           """Elige el nombre de función más probable entre las opciones válidas."""
           pass
       
       def extraer_numero(self, prompt_ids: list[int]) -> float:
           """Extrae un número usando decodificación restringida (solo dígitos y punto)."""
           pass
       
       def extraer_string(self, prompt_ids: list[int]) -> str:
           """Extrae una cadena usando decodificación restringida."""
           pass
   ```

### Hora 5: El orquestador (function_caller.py)

```python
class FunctionCaller:
    def __init__(self, modelo):
        self.modelo = modelo
        self.vocabulario = VocabIndex(modelo.get_path_to_vocabulary_json())
        self.decodificador = DecodificadorRestringido(modelo, self.vocabulario)
    
    def resolver(self, prompt: str, funciones: list[FunctionDefinition]) -> FunctionCall:
        """Dada una pregunta, elige función y extrae argumentos."""
        # 1. Construir prompt para elegir función
        # 2. Elegir función con decodificación restringida
        # 3. Para cada parámetro, extraer su valor
        # 4. Devolver FunctionCall
        pass
```

---

## 8. Ejemplos de código para practicar (NO son del proyecto)

### Ejemplo 1: Tokenizador simple

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

### Ejemplo 2: Diccionario token ↔ ID

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

### Ejemplo 3: Softmax y argmax

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

### Ejemplo 4: Máscara simple

```python
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

---

## 9. Preguntas frecuentes

### ¿Por qué no puedo usar transformers o pytorch?

Porque el proyecto quiere que aprendas a implementar la decodificación restringida desde cero, no que uses una librería que ya lo hace por ti.

### ¿Por qué no puedo usar heurísticas para elegir la función?

Porque el objetivo es que el **modelo** elija la función basándose en su comprensión del lenguaje, no tú con reglas manuales como "si la pregunta contiene 'suma', elige fn_sumar".

### ¿Qué es `llm_sdk`?

Es un **wrapper** (envoltorio) que te dan para interactuar con el modelo Qwen3-0.6B sin tener que instalar pytorch ni transformers. Es como un "control remoto" para el modelo.

### ¿Qué es `uv`?

Es un gestor de paquetes de Python más rápido que `pip`. Se usa así:
```bash
uv sync          # Instala dependencias
uv run python    # Ejecuta Python con el entorno virtual
```

### ¿Qué es Pydantic?

Es una librería que valida datos automáticamente. Si dices que un campo debe ser un string y le pasas un número, Pydantic lanza un error claro.

```python
from pydantic import BaseModel

class Usuario(BaseModel):
    nombre: str
    edad: int

usuario = Usuario(nombre="Ana", edad="veinte")  # Error: edad debe ser int
```

---

## 10. Checklist antes de empezar a programar

- [ ] Tengo Python 3.10+ instalado
- [ ] Tengo `uv` instalado
- [ ] He creado la estructura de carpetas
- [ ] He copiado `llm_sdk` en la raíz
- [ ] He creado `pyproject.toml` con las dependencias
- [ ] He creado el Makefile con las reglas obligatorias
- [ ] He creado los archivos de datos de entrada
- [ ] Entiendo qué es un token
- [ ] Entiendo qué son los logits
- [ ] Entiendo qué es la decodificación restringida
- [ ] He leído el subject completo

---

## 11. Siguiente paso

Cuando termines de preparar el entorno y crear los archivos básicos, empieza por:

1. **`models.py`**: Define las clases Pydantic
2. **`vocab.py`**: Carga el vocabulario y crea los diccionarios de búsqueda
3. **`constrained_decoder.py`**: Implementa la máscara y la selección de tokens
4. **`prompt_builder.py`**: Construye los prompts para el modelo
5. **`function_caller.py`**: Coordina todo el proceso
6. **`tools.py`**: Lee y escribe archivos JSON
7. **`__main__.py`**: El punto de entrada del programa

¡Mucha suerte! Recuerda: el objetivo no es terminar rápido, sino **entender** cada paso.
