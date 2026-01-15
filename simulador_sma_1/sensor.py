from typing import Dict, Any


class SensorPosicao:
    """
    Sensor simples: devolve a posição do agente e info básica do ambiente.
    """

    def __init__(self, alcance: int = 1):
        self.alcance = alcance

    def ler(self, ambiente, agente) -> Dict[str, Any]:
        # posição do agente (compatível com 'posicao' ou 'pos')
        if hasattr(agente, "posicao"):
            pos = agente.posicao
        elif hasattr(agente, "pos"):
            pos = agente.pos
        else:
            pos = (0, 0)

        return {
            "posicao": pos,
            "objetivos": list(getattr(ambiente, "objetivos", [])),
            "obstaculos": list(getattr(ambiente, "obstaculos", [])),
            "largura": getattr(ambiente, "largura", 0),
            "altura": getattr(ambiente, "altura", 0),
            "alcance": self.alcance,
        }