from typing import List, Any, Dict, Tuple, Optional
from ambiente import AmbienteBase
from agente import AgenteBase
from tqdm import trange
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Simulador:
    def __init__(self):
        self.ambiente: Optional[AmbienteBase] = None
        self.agentes: List[AgenteBase] = []

    @classmethod
    def cria(cls, ambiente: AmbienteBase, agentes: List[AgenteBase]):
        sim = cls()
        sim.ambiente = ambiente
        sim.agentes = agentes
        return sim

    def listaAgentes(self) -> List[AgenteBase]:
        return self.agentes

    def _obtem_observacao(self, agente: AgenteBase) -> Any:

        sensores = getattr(agente, 'sensores', None)
        single_sensor = getattr(agente, 'sensor', None)

        if sensores:
            obs = {}
            for i, s in enumerate(sensores):
                if hasattr(s, "ler"):
                    try:
                        obs_part = s.ler(self.ambiente, agente)
                    except Exception as e:
                        logger.debug(f"Sensor {s} falhou: {e}")
                        obs_part = None
                    obs[f"sensor_{i}_{s.__class__.__name__}"] = obs_part
            return obs

        if single_sensor and hasattr(single_sensor, "ler"):
            try:
                return single_sensor.ler(self.ambiente, agente)
            except Exception as e:
                logger.debug(f"Sensor único falhou: {e}")

        if hasattr(self.ambiente, "observacaoPara"):
            return self.ambiente.observacaoPara(agente)
        elif hasattr(self.ambiente, "observacao_para"):
            return self.ambiente.observacao_para(agente)
        else:
            raise AttributeError("Ambiente não implementa observacao_para e não há sensores")

    def _aplica_acao_no_ambiente(self, acao: Any, agente: AgenteBase) -> Tuple[float, bool]:
        if hasattr(self.ambiente, "agir"):
            # Mantém o formato do enunciado/professor: agir(accao, agente)
            result = self.ambiente.agir(acao, agente)
            if isinstance(result, tuple) and len(result) >= 2:
                recompensa, terminou = float(result[0]), bool(result[1])
            else:
                recompensa, terminou = float(result), False
            return recompensa, terminou
        raise AttributeError("Ambiente não possui método 'agir'")

    def executa(self, passos: int = 100, visualizar: bool = True, desconto: float = 0.99) -> Dict[str, Any]:
        if self.ambiente is None:
            raise RuntimeError("Simulador sem ambiente associado.")
        if not self.agentes:
            raise RuntimeError("Simulador sem agentes.")

        total_recompensa = 0.0
        recompensas_por_passo: List[float] = []
        passos_executados = 0

        for passo in trange(passos, desc="Simulação"):
            passos_executados += 1
            recompensa_este_passo = 0.0

            if visualizar:
                logger.info(f"\nPasso {passo + 1}/{passos}")

            obs_dict: Dict[AgenteBase, Any] = {}
            for agente in self.agentes:
                try:
                    obs = self._obtem_observacao(agente)
                except Exception as e:
                    logger.warning(f"Impossivel observação para agente {getattr(agente, 'nome', agente)}: {e}")
                    obs = None
                obs_dict[agente] = obs

            acoes_dict: Dict[AgenteBase, Any] = {}
            for agente, obs in obs_dict.items():
                action = None
                if hasattr(agente, "age"):
                    action = agente.age(obs)
                elif hasattr(agente, "agir"):
                    action = agente.agir(obs)
                elif hasattr(agente, "escolher_acao"):
                    action = agente.escolher_acao(obs)
                else:
                    raise AttributeError(f"Agente {agente} não tem método age/agir/escolher_acao")
                acoes_dict[agente] = action

            termino_global = False
            for agente, acao in acoes_dict.items():
                try:
                    recompensa, terminou = self._aplica_acao_no_ambiente(acao, agente)
                except Exception as e:
                    logger.error(f"Erro ao aplicar ação do agente {getattr(agente, 'nome', agente)}: {e}")
                    recompensa, terminou = 0.0, False


                if hasattr(agente, "avaliacaoEstadoAtual"):
                    try:
                        agente.avaliacaoEstadoAtual(recompensa)
                    except Exception:
                        logger.debug("avaliacaoEstadoAtual falhou, ignorado")

                if hasattr(agente, "regista_desempenho"):
                    try:
                        agente.regista_desempenho(obs_dict.get(agente), acao, recompensa)
                    except Exception:
                        logger.debug("regista_desempenho falhou, ignorado")

                recompensa_este_passo += float(recompensa)
                if terminou:
                    termino_global = True

            if hasattr(self.ambiente, "atualizacao"):
                try:
                    self.ambiente.atualizacao()
                except Exception:
                    logger.debug("atualizacao do ambiente falhou, ignorado")

            recompensas_por_passo.append(recompensa_este_passo)
            total_recompensa += recompensa_este_passo

            if visualizar:
                try:
                    self.imprimeAmbiente()
                except Exception:
                    logger.debug("imprimeAmbiente falhou")

            objetivos = getattr(self.ambiente, "objetivos", None)

            if isinstance(objetivos, (list, set)) and len(objetivos) == 0:
                if visualizar:
                    logger.info(f"\nObjetivos esgotados. Pára no passo {passos_executados}")
                break

            if hasattr(self.ambiente, "terminou") and callable(getattr(self.ambiente, "terminou")):
                try:
                    if self.ambiente.terminou():
                        if visualizar:
                            logger.info(f"\nAmbiente reportou que terminou no passo {passos_executados}")
                        break
                except Exception:
                    pass

            if termino_global:
                if visualizar:
                    logger.info(f"\nUm agente atingiu condição de término no passo {passos_executados}")
                break

        sucesso_count = 0

        for ag in self.agentes:
            try:

                if hasattr(self.ambiente, "verifica_objetivo_alcancado"):
                    if self.ambiente.verifica_objetivo_alcancado(ag):
                        sucesso_count += 1
                else:
                    pos = getattr(ag, "posicao", None)
                    if pos is not None and pos in getattr(self.ambiente, "objetivos", []):
                        sucesso_count += 1
            except Exception:
                pass

        recompensa_descontada = sum((desconto ** t) * r for t, r in enumerate(recompensas_por_passo))
        media_movel = []
        window = 5
        if len(recompensas_por_passo) >= window:
            media_movel = np.convolve(recompensas_por_passo, np.ones(window) / window, mode="valid").tolist()

        return {
            "recompensa_total": total_recompensa,
            "recompensa_media_por_agente": total_recompensa / max(1, len(self.agentes)),
            "recompensa_descontada": recompensa_descontada,
            "passos_executados": passos_executados,
            "sucesso_rate": sucesso_count / max(1, len(self.agentes)),
            "recompensas_passo_a_passo": recompensas_por_passo,
            "media_movel": media_movel
        }

    def imprimeAmbiente(self):
        print("Legenda: A=Agente, O=Objetivo, R=Recurso, X=Obstáculo\n")
        largura = getattr(self.ambiente, "largura", None)
        altura = getattr(self.ambiente, "altura", None)

        if largura is None or altura is None:
            grid = getattr(self.ambiente, "grid", None)
            if grid:
                altura = len(grid)
                largura = len(grid[0])
            else:
                print("Ambiente sem dimensão conhecida - impressão não disponivel")
                return

        objetivos = set(getattr(self.ambiente, "objetivos", []))
        obstaculos = set(getattr(self.ambiente, "obstaculos", []))
        recursos_attr = getattr(self.ambiente, "recursos", {})
        recursos = set(recursos_attr.keys()) if isinstance(recursos_attr, dict) else set(recursos_attr)

        agentes_pos = {}
        for ag in self.agentes:
            pos = getattr(ag, "posicao", None)
            if pos is not None:
                agentes_pos[pos] = ag

        for y in range(altura):
            linha = ""
            for x in range(largura):
                pos = (x, y)
                if pos in agentes_pos:
                    linha += "A "
                elif pos in objetivos:
                    linha += "O "
                elif pos in obstaculos:
                    linha += "X "
                elif pos in recursos:
                    linha += "R "
                else:
                    linha += ". "
            print(linha)
        print("\n")
