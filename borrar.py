from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Define la ruta local a tu carpeta del modelo
model_path = os.path.abspath("./llm_sdk/Small_LLM_Model")

# 2. Carga el tokenizador y el modelo desde esa ruta
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto" # Mueve el modelo automáticamente a la GPU si está disponible
)

# Preparar el texto de entrada
prompt = "Dame una idea para un desayuno saludable."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generar la respuesta
outputs = model.generate(**inputs, max_new_tokens=150)
respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(respuesta)