import pygame
from typing import List, Tuple

from simulador import Simulador


class VisualizadorPygame:
    def __init__(self, ambiente, agentes, tamanho_celula=80):
        pygame.init()

        self.ambiente = ambiente
        self.agentes = agentes
        self.tamanho_celula = tamanho_celula

        self.largura_px = ambiente.largura * tamanho_celula
        self.altura_px = ambiente.altura * tamanho_celula

        self.ecra = pygame.display.set_mode((self.largura_px, self.altura_px))
        pygame.display.set_caption('Simulador Pygame')

        self.clock = pygame.time.Clock()

        self.COR_FUNDO = (240, 240, 240)
        self.COR_GRELHA = (200, 200, 200)
        self.COR_AGENTE = (0, 100, 255)
        self.COR_OBJETIVO = (0, 200, 0)
        self.COR_OBSTACULO = (50, 50, 50)

    def desenhar_grelha(self):
        for x in range(self.ambiente.largura):
            for y in range(self.ambiente.altura):
                rect = pygame.Rect(x * self.tamanho_celula, y * self.tamanho_celula, self.tamanho_celula, self.tamanho_celula)
                pygame.draw.rect(self.ecra, self.COR_GRELHA, rect, 1)

    def desenhar_objetivo(self):
        for (x, y) in self.ambiente.objetivos:
            rect = pygame.Rect(x * self.tamanho_celula, y * self.tamanho_celula, self.tamanho_celula, self.tamanho_celula)
            pygame.draw.rect(self.ecra, self.COR_OBJETIVO, rect)

    def desenhar_obstaculo(self):
        for (x, y) in self.ambiente.obstaculos:
            rect = pygame.Rect(x * self.tamanho_celula, y * self.tamanho_celula, self.tamanho_celula, self.tamanho_celula)
            pygame.draw.rect(self.ecra, self.COR_OBSTACULO, rect)

    def desenhar_agentes(self):
        for ag in self.agentes:
            x, y = ag.posicao
            centro = (x*self.tamanho_celula + self.tamanho_celula // 2, y*self.tamanho_celula + self.tamanho_celula // 2, )
            pygame.draw.circle(self.ecra, self.COR_AGENTE, centro, self.tamanho_celula // 3)

    def atualizar(self, fps=120):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.ecra.fill(self.COR_FUNDO)
        self.desenhar_grelha()
        self.desenhar_objetivo()
        self.desenhar_obstaculo()
        self.desenhar_agentes()

        pygame.display.flip()
        self.clock.tick(fps)