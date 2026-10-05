# Sesión de trabajo: Call Me Maybe

## Fecha: 2026-10-05

## Progreso actual

### Completado:
- **Fase 0**: Entorno preparado (carpetas, pyproject.toml, Makefile, llm_sdk, datos)
- **Fase 1**: `src/models.py` creado (ParameterDefinition, FunctionDefinition, FunctionCall, TestPrompt)
- **Fase 6 (parcial)**: `src/tools.py` con `json_reader`, `load_function_def`, `load_prompt`

### En progreso:
- **Fase 6**: `src/tools.py` — falta `json_exporter`

### Pendiente:
- **Fase 2**: `src/vocab.py` — Necesita refactorización (actualmente es una versión simple)
- **Fase 3**: `src/constrained_decoder.py` — No existe
- **Fase 4**: `src/prompt_builder.py` — No existe
- **Fase 5**: `src/function_caller.py` — No existe
- **Fase 7**: `src/__main__.py` — Necesita actualizarse cuando existan los otros módulos
- **Fase 8**: Probar y depurar

## Archivos del proyecto:

```
src/
├── __init__.py          ← Solo docstring
├── __main__.py          ← Punto de entrada (carga archivos, ejecuta modelo, guarda resultados)
├── models.py            ← Clases Pydantic (correcto)
├── vocab.py             ← Versión simple, necesita refactorización
└── tools.py             ← json_reader, load_function_def, load_prompt (falta json_exporter)

data/input/
├── function_definitions.json   ← 5 funciones
└── function_calling_tests.json ← 1 prompt de prueba
```

## Notas importantes:

- El usuario prefiere que se le indique solo lo que hace mal, no lo que hace bien
- No revisar archivos a menos que se pida explícitamente
- El usuario quiere entender el "por qué" de cada cosa
- La guía está en `GUIDES/GUIA_CLASE.md`

## Próximos pasos:

1. Completar `json_exporter` en `src/tools.py`
2. Refactorizar `src/vocab.py` para que use el formato real del vocabulario del modelo
3. Crear `src/prompt_builder.py`
4. Crear `src/constrained_decoder.py`
5. Crear `src/function_caller.py`
6. Actualizar `src/__main__.py` si es necesario
7. Probar y depurar
