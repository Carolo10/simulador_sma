from abc import ABC, abstractmethod
import pickle
import math
import random
from typing import Dict, Tuple, Any, Optional, List


class AgenteBase(ABC):
    @abstractmethod
    def age(self, obs):
        """Escolhe a ação a partir da observação."""
        pass

    @abstractmethod
    def avaliacaoEstadoAtual(self, recompensa: float):
        """Atualiza informação de recompensa / desempenho."""
        pass

    @abstractmethod
    def instala(self, sensor):
        """Instala um sensor no agente."""
        pass

    @abstractmethod
    def comunica(self, mensagem: str, de_agente: "AgenteBase"):
        """Comunicação simples entre agentes."""
        pass


class Agente(AgenteBase):
    # ações possíveis numa grelha 2D
    ACOES = ["cima", "baixo", "esquerda", "direita", "parado"]

    def __init__(
        self,
        nome: str,
        modo: str = "test",      #learn  "test"
        tipo_politica: str = "qlearning",
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.nome = nome
        self.posicao: Tuple[int, int] = (0, 0)
        self.modo = modo        #  learn "teste"
        self.tipo_politica = tipo_politica  # "qlearning" ou "fixa"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tipo_politica = tipo_politica

        # Q-learning
        self.q_table: Dict[Any, Dict[str, float]] = {}
        self.last_state = None
        self.last_action = None

        # sensores
        self.sensor = None

        # desempenho / métricas
        self.recompensa_total = 0.0
        self.historico_passos: List[Tuple[int, int]] = []
        self.historico_distancias: List[float] = []
        self.historico_colisoes: int = 0
        self.historico_decisoes_erradas: int = 0
        self.tempo_inicio_ep: Optional[float] = None
        self.tempo_fim_ep: Optional[float] = None

    # ---------- fabrica simples ----------

    @classmethod
    def cria(cls, nome, modo: str = "test", tipo_politica="qlearning"):
        return cls(nome=f"Agente_{nome}", modo=modo, tipo_politica=tipo_politica)

    # ---------- interface de base ----------

    def instala(self, sensor):
        self.sensor = sensor

    def comunica(self, mensagem: str, de_agente: "AgenteBase"):
        print(f"{de_agente.nome} -> {self.nome}: {mensagem}")

    # ---------- Q-learning ----------

    def _estado_from_obs(self, obs) -> Any:
        """
        Converte uma observação no hashable mais simples sem muita coisa
        Aqui usamos:
          - posição do agente
          - posição do primeiro objetivo normalmente a ultima
        """
        if isinstance(obs, tuple):
            return obs

        pos = obs.get("posicao_agente", obs.get("posicao", self.posicao))
        objetivos = obs.get("objetivos", [])
        obj = objetivos[0] if objetivos else None

        x,y = pos
        obstaculos = set(obs.get("obstaculos", []))
        vizinhanca = (
        (x, y-1) in obstaculos,
        (x, y+1) in obstaculos,
        (x-1,y) in obstaculos,
        (x+1, y) in obstaculos,
        )

        return (pos, obj, vizinhanca)

    def _init_state(self, estado):
        if estado not in self.q_table:
            self.q_table[estado] = {a: 0.0 for a in self.ACOES}

    def _escolhe_acao(self, estado):
        self._init_state(estado)

        # em modo teste não há exploração zero epsilon zero exploraçao
        eps = 0.0 if self.modo == "test" else self.epsilon

        if random.random() < eps:
            return random.choice(self.ACOES)


        # desempate aleatório para nao ficar parado se os valores forem iguais feito para evitar ficar parado
        acoes_estado = self.q_table[estado]
        melhor_q = max(acoes_estado.values())
        melhores = [a for a, q in acoes_estado.items() if q == melhor_q]
        return random.choice(melhores)

        # mudei isto, acho que fica melhor assim ,return max(acoes_estado, key=acoes_estado.get)

    def age(self, obs):
        """Escolha da ação de acordo com a politica, guarda o estado e devolve a ação."""
        if self.tipo_politica == "fixa":
            return self._acao_fixa(obs)
        else:
            #qlearning
            estado = self._estado_from_obs(obs)
            acao = self._escolhe_acao(estado)
            self.last_state = estado
            self.last_action = acao
            return acao

    def _acao_fixa(self, obs):
        "mover sempre na direção do objetivo mais proximo, pode ficar preso sem conseguir voltar para tras e tentar outro caminho"
        pos = obs.get("posicao_agente", self.posicao)
        objetivos = obs.get("objetivos", [])
        obstaculos = set(obs.get("obstaculos", []))
        if not objetivos:
            return "parado"
        obj = objetivos[0]

        movimentos = {
            "cima": (pos[0], pos[1] - 1),
            "baixo": (pos[0], pos[1] + 1),
            "esquerda": (pos[0] - 1, pos[1]),
            "direita": (pos[0] + 1, pos[1]),
        }

        movimentos_validos = {
            acao: nova_pos
            for acao, nova_pos in movimentos.items()
            if 0 <= nova_pos[0] < obs["largura"]
               and 0 <= nova_pos[1] < obs["altura"]
               and nova_pos not in obstaculos
        }

        if not movimentos_validos:
            return "parado"

        melhor_dist = abs(pos[0] - obj[0]) + abs(pos[1] - obj[1])
        acoes_possiveis = [
            acao
            for acao, nova_pos in movimentos_validos.items()
            if abs(nova_pos[0] - obj[0]) + abs(nova_pos[1] - obj[1]) < melhor_dist
        ]

        if not acoes_possiveis:
            acoes_possiveis = list(movimentos_validos.keys())

        return random.choice(acoes_possiveis)


    def update_transition(self, next_obs, recompensa: float, terminou: bool):
        """Atualização da Q-table segundo a transição (s, a, r, s')
        alpha > taxa de aprendizagem
         gamma > desconto do futuro
         r > recompensa recebida
         s' > próximo estado"""
        if self.modo != "learn":
            return
        if self.last_state is None or self.last_action is None:
            return

        estado = self.last_state
        acao = self.last_action
        prox_estado = self._estado_from_obs(next_obs)

        self._init_state(estado)
        self._init_state(prox_estado)

        q_atual = self.q_table[estado][acao]
        max_q_prox = max(self.q_table[prox_estado].values())

        novo_q = q_atual + self.alpha * (recompensa + self.gamma * max_q_prox - q_atual)
        self.q_table[estado][acao] = novo_q

        # annealing do epsilon vai aumentando e assim vai explorar cada vez menos e usar mais do conhecimento que já tem
        self.epsilon = max(0.01, self.epsilon * 0.995)

        if terminou:
            self.last_state = None
            self.last_action = None

    # ---------- métricas ----------

    def avaliacaoEstadoAtual(self, recompensa: float):
        self.recompensa_total += recompensa

    def regista_passos(
        self,
        acao: Optional[str],
        pos_antiga: Optional[Tuple[int, int]],
        pos_nova: Optional[Tuple[int, int]],
        objetivo: Optional[Tuple[int, int]] = None,
    ):
        if pos_nova is not None:
            self.historico_passos.append(pos_nova)

            if objetivo is not None:
                dist = abs(pos_nova[0] - objetivo[0]) + abs(pos_nova[1] - objetivo[1])
                self.historico_distancias.append(dist)

        # colisão simples, tentou mover e ficou no mesmo sítio
        if (
            pos_antiga is not None
            and pos_nova is not None
            and pos_antiga == pos_nova
            and acao in ["cima", "baixo", "esquerda", "direita"]
        ):
            self.historico_colisoes += 1

        if acao is None:
            self.historico_decisoes_erradas += 1

    def regista_desempenho(self, obs, acao, recompensa):
        # info extra se for para dar debug mas não é preciso agora mais tarde talvez
        pass

    def calculo_metricas(self, objetivo: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        recompensa_total = self.recompensa_total
        passos_total = len(self.historico_passos)

        media_recompensa_por_passo = (
            recompensa_total / passos_total if passos_total > 0 else 0.0
        )

        # sucesso se alguma das posições for igual à  do bjetivo
        # sucesso: verifica se o agente terminou no objetivo
        if objetivo is not None:
            taxa_sucesso = 1.0 if self.posicao == objetivo else 0.0
        else:
            taxa_sucesso = 0.0

        if self.tempo_inicio_ep is not None and self.tempo_fim_ep is not None:
            tempo_medio = self.tempo_fim_ep - self.tempo_inicio_ep
        else:
            tempo_medio = 0.0

        distancia_media = (
            sum(self.historico_distancias) / len(self.historico_distancias)
            if self.historico_distancias
            else 0.0
        )

        return {
            "recompensa_total": recompensa_total,
            "passos_total": passos_total,
            "media_recompensa_por_passo": media_recompensa_por_passo,
            "taxa_sucesso": taxa_sucesso,
            "tempo_medio_por_episodio": tempo_medio,
            "colisoes": self.historico_colisoes,
            "distancia_media_ao_objetivo": distancia_media,
            "decisoes_erradas": self.historico_decisoes_erradas,
        }

    # ---------- utilidades ----------
    #carregar as tabelas
    def reset_recompensa(self):
        self.recompensa_total = 0.0

    def guardar_q_table(self, ficheiro="q_table.pkl"):
        with open(ficheiro, "wb") as f:
            pickle.dump(self.q_table, f)

    def carregar_q_table(self, ficheiro="q_table.pkl"):
        try:
            with open(ficheiro, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}
