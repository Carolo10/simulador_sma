from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
import random


class AmbienteBase(ABC):
    @abstractmethod
    def observacaoPara(self, agente):
        pass

    @abstractmethod
    def agir(self, acao, agente):
        """Aplica a ação no ambiente e devolve (recompensa, terminou)."""
        pass

    @abstractmethod
    def atualizacao(self):
        pass

    @abstractmethod
    def verifica_objetivo_alcancado(self, agente) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass


class Ambiente(AmbienteBase):
    """
    Ambiente genérico em grelha para Farol e Labirinto, não fazemos diferenciação de momento
    Suporta:
      - vários agentes, mas não testado ainda como deve ser
      - objetivos
      - obstáculos
     recompensa foi melhorada para incentivar movimento e aproximação ao objetivo sem ter de ficar parado para nao perder pontos como fazia antes
    """

    def __init__(self, largura: int, altura: int, max_passos: int = 50):
        self.largura = largura
        self.altura = altura
        self.max_passos = max_passos

        self.agentes: List = []
        self.objetivos: List[Tuple[int, int]] = []
        self.obstaculos: List[Tuple[int, int]] = []

        self.passos = 0
        self._posicoes_iniciais: Dict[object, Tuple[int, int]] = {}

    # ---------- configuração ----------

    def adicionaAgente(self, agente, posicao: Tuple[int, int]):
        agente.posicao = posicao
        self.agentes.append(agente)
        self._posicoes_iniciais[agente] = posicao

    def adicionaObjetivo(self, posicao: Tuple[int, int]):
        if posicao not in self.objetivos:
            self.objetivos.append(posicao)

    def adicionaObstaculo(self, posicao: Tuple[int, int]):
        if posicao not in self.obstaculos:
            self.obstaculos.append(posicao)

    # ---------- ciclo ----------

    def observacaoPara(self, agente):
        """
        Observação simples:
          - posição atual
          - lista de objetivos
          - lista de obstáculos
          - dimensões da grelha
        """
        return {
            "posicao_agente": agente.posicao,
            "objetivos": list(self.objetivos),
            "obstaculos": list(self.obstaculos),
            "largura": self.largura,
            "altura": self.altura,
        }

    def _proxima_posicao(self, pos: Tuple[int, int], acao: str) -> Tuple[int, int]:
        """
        Calcula a próxima posição na grelha dada a ação.
        Ações válidas: "cima", "baixo", "esquerda", "direita".
        Qualquer outra ação mantém a posição.
        """
        x, y = pos
        if acao == "cima":
            y = max(0, y - 1)
        elif acao == "baixo":
            y = min(self.altura - 1, y + 1)
        elif acao == "esquerda":
            x = max(0, x - 1)
        elif acao == "direita":
            x = min(self.largura - 1, x + 1)
        # se tiver parado ou ação desconhecida fica igual
        return x, y

    def _distancia_objetivo_mais_proximo(self, pos: Tuple[int, int]) -> Optional[int]:
        """
        Distância  ao objetivo mais próximo.
        Se não houver objetivos, devolve nenhum.
        """
        if not self.objetivos:
            return None
        x, y = pos
        return min(abs(x - ox) + abs(y - oy) for (ox, oy) in self.objetivos)

    def agir(self, acao, agente):
        """
        Movimento na grelha com:
          - custo base por passo: -0.01 quantos mais der mais perde
          - penalização extra se ficar parado (antes ele não andava porque era mais vantajoso ficar parado)
          - recompensa por se aproximar do objetivo
          - penalização por se afastar do objetivo
          - colisão com obstáculo: -1.0 e fica no sítio
          - chegar a objetivo: +1.0 e termina episódio

        Isto incentiva o agente a mexer-se e a procurar o objetivo

        """
        self.passos += 1

        pos_atual = agente.posicao
        dist_antes = self._distancia_objetivo_mais_proximo(pos_atual)

        # calcula posição tentada
        nova_pos = self._proxima_posicao(pos_atual, acao)

        recompensa = -0.01  # custo base de existir um passo
        terminou = False

        # colisão com obstáculo: fica no sítio e penaliza
        if nova_pos in self.obstaculos:
            nova_pos = pos_atual
            recompensa -= 1.0  # penalização mais forte por bater em obstáculo

        # atualiza posição do agente
        agente.posicao = nova_pos

        # distância depois do movimento (já aplicadas colisões)
        dist_depois = self._distancia_objetivo_mais_proximo(nova_pos)

        # penalização extra se não sair do sítio (sem ser por objetivo)
        if nova_pos == pos_atual:
            recompensa -= 0.2  # ficar parado é pior que dar um passo

        # incentivo/penalização por aproximação ao(s) objetivo(s)
        if dist_antes is not None and dist_depois is not None:
            if dist_depois < dist_antes:
                recompensa += 0.1  # aproximou-se do objetivo
            elif dist_depois > dist_antes:
                recompensa -= 0.1  # afastou-se do objetivo

        # recompensa adicional por chegar ao objetivo
        if nova_pos in self.objetivos:
            recompensa += 1.0
            terminou = True

        # termina se rebentar o limite de passos
        if self.passos >= self.max_passos:
            terminou = True

        return recompensa, terminou

    def atualizacao(self):
        """
        Neste projeto, o ambiente não tem dinâmica própria
        não se mexem objetivos nem obstáculos entre passos.
        """
        pass

    def verifica_objetivo_alcancado(self, agente) -> bool:
        return agente.posicao in self.objetivos

    def reset(self):
        """Volta a colocar os agentes nas posições iniciais e zera o contador de passos."""
        self.passos = 0
        for ag, pos_ini in self._posicoes_iniciais.items():
            ag.posicao = pos_ini
