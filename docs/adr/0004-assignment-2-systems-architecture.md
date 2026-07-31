# 4. Arquitetura do Assignment 2: Systems (Triton RMSNorm, DDP, Sharded Optimizer e Benchmarking)

* **Data:** 2026-07-31
* **Status:** Aceito

## Contexto
O treinamento e inferência de Large Language Models (LLMs) em larga escala exigem otimizações profundas no nível de software de sistemas. O Assignment 2 do Stanford CS336 foca no desenvolvimento de técnicas fundamentais de aceleração e paralelismo:
1. Fusão de kernels customizados de hardware (Triton) para reduzir overhead de leitura/escrita em memória global da GPU (HBM).
2. Paralelismo de dados distribuído (*Distributed Data Parallel* - DDP) para sincronização eficiente de gradientes em múltiplas GPUs/processos.
3. Particionamento de estados de otimizador (*Optimizer State Sharding* / ZeRO-1) para eliminar redundâncias de memória durante a otimização.
4. Utilitários de *benchmarking* e *profiling* para mensurar latência, throughput e gargalos de execução.

## Decisão

### 1. Fused RMSNorm em Triton com Fallback Transparente
- Implementaremos a camada `RMSNorm` fundindo as operações de cálculo do valor quadrático médio (RMS), divisão de escala e multiplicação de ganho ponderado em um único kernel Triton.
- **Fallback de Portabilidade:** Como o ecossistema Windows/CPU não possui suporte nativo ao compilador Triton CUDA, o módulo detectará dinamicamente a disponibilidade do CUDA/Triton. Quando indisponível, utilizará um *fallback* numérico exato em PyTorch nativo autograd, preservando a interface de uso e garantindo que toda a suíte de testes seja executada em qualquer plataforma.

### 2. Distributed Data Parallel (DDP)
- Implementaremos uma classe `DDP` que encapsula qualquer `nn.Module`.
- A sincronização de gradientes será feita via `all_reduce` utilizando o `torch.distributed`.
- Para maximizar a vazão de rede, suportaremos *bucketing* (agrupamento de gradientes em buffers contíguos) e registro de *hooks* no backward pass para sobrepor comunicação de rede com computação das camadas anteriores.

### 3. Sharded Optimizer (ZeRO-1)
- Implementaremos o `ShardedOptimizer` particionando os estados do otimizador (como m1/m2 do AdamW) entre os ranks distribuídos.
- Cada rank atualiza exclusivamente a sua fração de parâmetros e espalha os novos valores via `all_gather` para manter a consistência dos pesos.

### 4. Suíte de Benchmarking e Profiling
- Desenvolveremos scripts em `src/cs336_systems/benchmark.py` para medição precisa de tempo por passo (latência em ms), tokens/segundo (throughput) e memória alocada, além de integração com `torch.profiler`.

## Consequências
- Código altamente modular e resiliente a diferentes plataformas (Windows/Linux/CPU/CUDA).
- Cobertura completa de testes unitários para verificação numérica estrita em relação ao PyTorch de referência.
- Conformidade total com as diretrizes do Stanford CS336 e do repositório.
