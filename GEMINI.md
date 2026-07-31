# Diretrizes de Desenvolvimento e Regras de Git para o Projeto

## Regras de Commit e Versionamento

1. **Commit Obrigatório para Qualquer Alteração:**
   - Toda e qualquer alteração no projeto (código, testes, documentação, ADRs, configurações) deve ser commitada.

2. **Granularidade Máxima de Commits (Commits Atômicos):**
   - As alterações devem ser divididas no maior nível de granularidade possível.
   - Cada commit deve representar uma única mudança lógica ou unidade de trabalho individual (ex: criação de um módulo, adição de um teste específico, atualização de uma ADR, etc.). Evite agrupar alterações não relacionadas em um único commit.

3. **Padrão de Nomenclatura (Conventional Commits):**
   - Os commits devem seguir o padrão de mensagens claras e descritivas usando os prefixos da convenção Conventional Commits:
     - `feat:` para novas funcionalidades ou componentes.
     - `fix:` para correção de bugs ou ajustes de problemas.
     - `docs:` para atualizações ou criações de documentação, comentários e ADRs.
     - `test:` para criação ou modificação de scripts e suítes de teste.
     - `chore:` para alterações em arquivos de configuração, dependências ou tarefas administrativas.
     - `refactor:` para refatorações que não alteram a funcionalidade existente.
     - `style:` para ajustes de formatação e linting.
     - `perf:` para otimizações de desempenho.

4. **Clareza nas Mensagens:**
   - O título e a descrição do commit devem explicar claramente o que foi feito e o motivo da alteração.

5. **Push Automático:**
   - Após a realização de commits, deve ser efetuado o `git push` imediatamente para sincronizar com o repositório remoto.
