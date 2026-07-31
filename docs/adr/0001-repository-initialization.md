# ADR 0001: Inicialização do Repositório e Estrutura Base

* **Status:** Aceito
* **Data:** 2026-07-31

## Contexto

Necessidade de estruturar o projeto `Stanford-CS336` para suportar o desenvolvimento de modelos de linguagem a partir do zero, seguindo boas práticas de desenvolvimento, testes automatizados e documentação técnica.

## Decisão

1. **Estrutura de Pastas:**
   - `src/`: Contém os módulos de código Python.
   - `tests/`: Contém testes automatizados.
   - `docs/`: Guarda documentações e ADRs (`docs/adr/`).
   - `tmp/`: Diretório temporário (ignorado pelo Git via `.gitignore`).

2. **Gerenciamento de Controle de Versão:**
   - Git como VCP primário com branch principal `main`.
   - Inclusão de `.gitignore` abrangente para ambiente Python e arquivos temporários.

## Consequências

- Facilidade no rastreamento de decisões de arquitetura e infraestrutura.
- Garantia de isolamento de arquivos temporários e dependências locais.
