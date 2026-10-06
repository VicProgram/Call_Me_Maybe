import json


class VocabIndex:
    def __init__(self, json_path: str):
        self.json_path = json_path
        data = self.load_json(self.json_path)

    def load_json(self, json_path: str):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception:
            raise FileNotFoundError("JSON not founded")

        return data

    def detect_key(self, data: dict):
        first_key = next(iter(data))

        if (type = int(first_key) == True):
            return type
        else:
            type = str(first_key)
            return type
            


    # region
     
    # def cargar_json(self):
    #     # Leer el archivo JSON
    #     pass
    
    # def buscar_exacto(self, token: str) -> list[int]:
    #     # Buscar un token exacto
    #     pass
    
    # def buscar_prefijo(self, prefijo: str) -> list[int]:
    #     # Buscar tokens que empiezan por un prefijo
    #     pass
    
    # def buscar_caracteres(self, caracteres: str) -> list[int]:
    #     # Buscar tokens compuestos solo por ciertos caracteres
    #     pass

    # endregion
