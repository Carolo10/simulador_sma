import os
import csv
import time
import random

import matplotlib.pyplot as plt

from ambiente import Ambiente
from agente import Agente
from sensor import SensorPosicao
from simulador import Simulador
from visualizador import VisualizadorPygame
from collections import deque


# --------------------------------------------------------------
#   Ciclo principal de episódios + métricas
# --------------------------------------------------------------

def executar_experiencia(
    ambiente: Ambiente,
    agentes,
    episodios: int = 5,
    passos_por_episodio: int = 10,
    visualizar: bool = False,
    penalizar_revisitas: bool = False,
):

    historico = []
    sim = Simulador.cria(ambiente, agentes)

    visualizador = None
    if visualizar:
        visualizador = VisualizadorPygame(ambiente, agentes)

    for ep in range(episodios):
        print(f"\nEpisódio {ep + 1}/{episodios}")

        # reset ao ambiente e posições iniciais
        ambiente.reset()
        # objetivo principal (usamos o primeiro)
        objetivo_principal = ambiente.objetivos[0] if ambiente.objetivos else None

        # reset aos agentes
        for ag in agentes:
            ag.reset_recompensa()
            ag.historico_passos = []
            ag.historico_distancias = []
            ag.historico_colisoes = 0
            ag.historico_decisoes_erradas = 0
            ag.tempo_inicio_ep = time.time()
            ag.tempo_fim_ep = None
            ag.last_state = None
            ag.last_action = None
            if penalizar_revisitas:
                ag._visit_count = {}

        terminou_global = False

        for passo in range(passos_por_episodio):
            for ag in agentes:
                # 1) Observação
                if ag.sensor is not None:
                    obs = ag.sensor.ler(ambiente, ag)
                else:
                    obs = ambiente.observacaoPara(ag)

                pos_antiga = getattr(ag, "posicao", None)

                # 2) Escolha da ação
                acao = ag.age(obs)

                # 3) Ação no ambiente
                recompensa, terminou = ambiente.agir(acao, ag)
                # --- penalização por revisitar posições (só labirinto)
                if penalizar_revisitas:
                    pos_atual = getattr(ag, "posicao", None)
                    if pos_atual is not None:
                        vc = ag._visit_count
                        vc[pos_atual] = vc.get(pos_atual, 0) + 1

                        repeticoes = vc[pos_atual] - 1
                        if repeticoes > 0:
                            pen = 0.06 * (repeticoes ** 2)
                           # if pen > 0.5:
                            #    pen = 0.5
                            recompensa -= pen
                # -----------

                # 4) Atualizar recompensa
                ag.avaliacaoEstadoAtual(recompensa)

                # 5) Nova observação para update da Q-table
                next_obs = ambiente.observacaoPara(ag)
                ag.update_transition(next_obs, recompensa, terminou)

                # 6) Registo de passos / distâncias / colisões
                pos_nova = getattr(ag, "posicao", None)
                ag.regista_passos(acao, pos_antiga, pos_nova, objetivo_principal)

                if terminou:
                    terminou_global = True

            # atualização de ambiente (se tiver dinâmica, ainda não)
            ambiente.atualizacao()

            if visualizar and visualizador is not None:
                visualizador.atualizar(fps=120)
                time.sleep(0.002)

            if terminou_global:
                print(f"  Episódio terminou no passo {passo + 1}")
                break

        # fim do episódio > métricas por agente
        metricas_ep = {}
        for ag in agentes:
            ag.tempo_fim_ep = time.time()
            metricas = ag.calculo_metricas(objetivo_principal)
            metricas_ep[ag.nome] = metricas
            print(f"[Ep {ep + 1}] Métricas {ag.nome}: {metricas}")

        historico.append(metricas_ep)

    return historico


# --------------------------------------------------------------
#   Gráficos e CSV
# --------------------------------------------------------------

def mostrar_curva_aprendizagem(
    historico,
    titulo: str,
    ficheiro_prefix: str,
    agente_nome: str,
):
    os.makedirs("resultados", exist_ok=True)

    # assumimos um agente (ou escolhemos pelo nome)
    medias = [ep[agente_nome]["media_recompensa_por_passo"] for ep in historico]
    sucessos = [ep[agente_nome]["taxa_sucesso"] for ep in historico]
    episodios = list(range(1, len(historico) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(episodios, medias, marker="o", label="Recompensa média por passo")
    plt.plot(episodios, sucessos, marker="x", label="Taxa de sucesso")
    plt.xlabel("Episódio")
    plt.ylabel("Valor")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)

    png_path = os.path.join("resultados", f"{ficheiro_prefix}_curva.png")
    plt.savefig(png_path)
    plt.close()

    # CSV: linha por agente / episódio
    csv_path = os.path.join("resultados", f"{ficheiro_prefix}_historico.csv")
    with open(csv_path, "w", newline="") as f:
        # assumimos que há pelo menos um episódio e um agente
        primeiro_ep = historico[0]
        primeiro_agente_nome = list(primeiro_ep.keys())[0]
        fieldnames = ["episodio", "agente"] + list(primeiro_ep[primeiro_agente_nome].keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, ep in enumerate(historico, start=1):
            for nome_agente, mets in ep.items():
                row = {"episodio": i, "agente": nome_agente}
                row.update(mets)
                writer.writerow(row)

    print(f"Gráfico e CSV guardados em: {png_path}, {csv_path}")


# --------------------------------------------------------------
#   Experiências
# --------------------------------------------------------------

#definir se existe caminho disponivel atraves do BFS
def existe_caminho(largura, altura, inicio, objetivo, obstaculos):
    obstaculos=set(obstaculos)
    visitados = set([inicio])
    fila = deque([inicio])

    while fila:
        x,y = fila.popleft()
        if(x,y) == objetivo:
            return True

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)

            if (
                0 <= nx < largura
                and 0 <= ny < altura
                and pos not in obstaculos
                and pos not in visitados
            ):
                visitados.add(pos)
                fila.append(pos)
    return False

def gerar_labirinto_valido(ambiente, inicio=(0,0)):
    largura = ambiente.largura
    altura = ambiente.altura
    objetivo = ambiente.objetivos[0]

    vizinhos_inicio = {
        inicio,
        (inicio[0] + 1, inicio[1]),
        (inicio[0] - 1, inicio[1]),
        (inicio[0], inicio[1] + 1),
        (inicio[0], inicio[1] - 1),
    }

    while True:
        obstaculos = set()
        n_obs = random.randint(3, 6)

        while len(obstaculos) < n_obs:
            pos = (
                random.randint(0, largura-1),
                random.randint(0, altura-1),
            )
            if (pos not in vizinhos_inicio and pos != objetivo
            ):
                obstaculos.add(pos)

        if existe_caminho(
            largura, altura, inicio, objetivo, obstaculos
        ):
            ambiente.obstaculos.clear()
            for o in obstaculos:
                ambiente.adicionaObstaculo(o)
            return


def experiencia_farol(tipo_politica="qlearning"):
    print("=== Experiência Farol ===")

    W, H = 10, 10   # <<< define aqui o tamanho

    ambiente = Ambiente(W, H, max_passos=30)
    ambiente.adicionaObjetivo((W-1, H-1))

    # obstáculos
    ambiente.adicionaObstaculo((2, 2))
    ambiente.adicionaObstaculo((2, 3))
    ambiente.adicionaObstaculo((3, 2))
    ambiente.adicionaObstaculo((3, 4))
    ambiente.adicionaObstaculo((7, 6))
    ambiente.adicionaObstaculo((8, 6))
    ambiente.adicionaObstaculo((7, 7))
    ambiente.adicionaObstaculo((8, 3))
    ambiente.adicionaObstaculo((4, 8))





    # treino
    agente = Agente.cria("farol", modo="learn", tipo_politica=tipo_politica)
    agente.q_table = {}
    agente.instala(SensorPosicao())
    ambiente.adicionaAgente(agente, (0, 0))

    historico = executar_experiencia(
        ambiente, [agente],
        episodios=45,
        passos_por_episodio=24,
        visualizar=False
    )

    # só guarda Q-table se for Q-learning1
    if tipo_politica == "qlearning":
        agente.guardar_q_table("qtable_farol.pkl")

    # prefixo diferente para não sobrescrever ficheiros
    prefix = "farol_qlearning" if tipo_politica == "qlearning" else "farol_fixa"
    mostrar_curva_aprendizagem(historico, f"Farol - Aprendizagem ({tipo_politica})", prefix, agente.nome)

    # teste
    agente_teste = Agente.cria("farol", modo="test", tipo_politica=tipo_politica)
    if tipo_politica == "qlearning":
        agente_teste.carregar_q_table("qtable_farol.pkl")
    agente_teste.instala(SensorPosicao())
    ambiente_teste = Ambiente(W, H, max_passos=30)
    ambiente_teste.adicionaObjetivo((W-1, H-1))
    ambiente_teste.adicionaObstaculo((2, 2))
    ambiente_teste.adicionaObstaculo((2, 3))
    ambiente_teste.adicionaObstaculo((3, 2))
    ambiente_teste.adicionaObstaculo((3, 4))
    ambiente_teste.adicionaObstaculo((7, 6))
    ambiente_teste.adicionaObstaculo((8, 6))
    ambiente_teste.adicionaObstaculo((7, 7))
    ambiente_teste.adicionaObstaculo((8, 3))
    ambiente_teste.adicionaObstaculo((4, 8))


    ambiente_teste.adicionaAgente(agente_teste, (0, 0))

    print("\n=== Fase de teste (Farol) ===")
    executar_experiencia(ambiente_teste, [agente_teste], episodios=13, passos_por_episodio=30, visualizar=True)

MAPA_LAB_10x10 = [
    (2, 0), (4, 0), (7, 0), (8, 0), (9, 0),
    (6, 1),
    (0, 2), (1, 2), (2, 2), (5, 2), (8, 2),
    (4, 3), (7, 3),
    (1, 4), (2, 4), (4, 4), (5, 4), (8, 4),
    (4, 5), (8, 5),
    (2, 6), (3, 6), (6, 6), (7, 6), (8, 6),
    (5, 7),
    (1, 8), (2, 8), (3, 8), (4, 8), (7, 8), (8, 8), (9, 8),
    (3, 9),
]

MAPAS_LABIRINTO = [MAPA_LAB_10x10]
# MAPA2 (placeholder) — quando quiseres, adicionas aqui outro mapa na lista acima


def aplicar_mapa_labirinto(ambiente: Ambiente):
    ambiente.obstaculos.clear()
    mapa = random.choice(MAPAS_LABIRINTO)
    for pos in mapa:
        ambiente.adicionaObstaculo(pos)



def experiencia_labirinto(tipo_politica: str):
    print("=== Experiência Labirinto ===")


    W, H = 10, 10  # <<< define aqui o tamanho

    ambiente = Ambiente(W, H, max_passos=1000)
    ambiente.adicionaObjetivo((W - 1, H - 1))

    aplicar_mapa_labirinto(ambiente)

    agente = Agente.cria("labirinto", modo="learn", tipo_politica=tipo_politica)
    agente.q_table = {}
    agente.last_state = None
    agente.last_action = None
    agente.instala(SensorPosicao())
    ambiente.adicionaAgente(agente, (0, 0))

    #Treino
    historico = executar_experiencia(
        ambiente,
        [agente],
        episodios=35,
        passos_por_episodio=400,
        visualizar=True,
        penalizar_revisitas=True,
    )

    mostrar_curva_aprendizagem(
        historico, "Labirinto - Aprendizagem", "labirinto", agente.nome
    )

    agente.guardar_q_table("qtable_labirinto.pkl")

    agente_teste = Agente.cria("labirinto", modo="test", tipo_politica=tipo_politica)
    agente_teste.carregar_q_table("qtable_labirinto.pkl")
    agente_teste.instala(SensorPosicao())
    ambiente_teste = Ambiente(W, H, max_passos=100)
    ambiente_teste.adicionaObjetivo((W - 1, H - 1))
    aplicar_mapa_labirinto(ambiente_teste)

    ambiente_teste.adicionaAgente(agente_teste, (0, 0))

    print("\n=== Fase de teste (Labirinto) ===")
    executar_experiencia(
        ambiente_teste,
        [agente_teste],
        episodios=5,
        passos_por_episodio=40,
        visualizar=True,
        penalizar_revisitas=True,
    )


# --------------------------------------------------------------
#   main
# --------------------------------------------------------------

def main():
    print("Simulador Multi-Agente")
    print("1 - Farol")
    print("2 - Labirinto")
    print("0 - Ambas")

    escolha = input("Escolha a experiência: ")

    print("Tipo de politica:")
    print("1 - Politica fixa")
    print("2 - Q-learning")
    politica = input("Escolha a política")

    tipo_politica = "fixa" if politica == "1" else "qlearning"

    if escolha == "1":
        experiencia_farol(tipo_politica)
    elif escolha == "2":
        experiencia_labirinto(tipo_politica)
    else:
        experiencia_farol(tipo_politica)
        experiencia_labirinto(tipo_politica)


if __name__ == "__main__":
    main()
