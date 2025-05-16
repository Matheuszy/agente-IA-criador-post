# Codex Post Generator: Seu Sistema Inteligente de Criação de Conteúdo para Redes Sociais 🚀

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d2410ef78f9a6c173d461700863273dc/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Google Gemini](https://ai.google.dev/static/images/favicon_ai.ico)](https://ai.google.dev/)

**Apresentamos o Codex Post Generator, uma solução inovadora para a criação de posts de tendências para redes sociais, impulsionada pela inteligência artificial do Google Gemini. Desenvolvido pela Codex System, uma startup focada em soluções Python para diversos nichos de mercado, este sistema utiliza uma sequência de agentes inteligentes para pesquisar, planejar, redigir e revisar conteúdo de forma interativa e eficaz.**

## 💡 Uma Solução Inteligente para Conteúdo Impactante

O Codex Post Generator simplifica o processo de criação de posts, permitindo que você explore tendências e gere conteúdo relevante para o seu público-alvo. O sistema é composto por quatro agentes especializados:

* **Agente Buscador:** Um especialista em identificar as últimas tendências e novidades relevantes para diversos nichos de mercado (marketing digital, jurídico, saúde, varejo, restaurantes, logística, contabilidade, finanças e imobiliárias). Utiliza o Google Search para encontrar informações atuais e impactantes, focando em dores reais e cases de sucesso.
* **Agente Planejador:** Com base nos lançamentos e tendências encontradas, este agente planeja os pontos mais relevantes a serem abordados no post, sempre com foco nos benefícios e vantagens para o nicho escolhido, evitando detalhes técnicos de implementação.
* **Agente Redator:** Um criativo especialista em gerar rascunhos de posts virais para o Instagram. Utiliza o plano fornecido para escrever conteúdo engajador, informativo e com linguagem simples, incluindo hashtags relevantes.
* **Agente Revisor:** Um editor meticuloso responsável por revisar o rascunho do post, verificando clareza, concisão, correção e tom adequado para o público-alvo específico de cada nicho.

## ✨ Funcionalidades Principais

* **Fluxo de Criação Interativo:** O usuário pode guiar o processo de criação do post, escolhendo qual etapa executar.
* **Pesquisa Inteligente de Tendências:** O Agente Buscador foca em novidades atuais e relevantes para diversos mercados.
* **Planejamento Estratégico de Conteúdo:** O Agente Planejador direciona o conteúdo para os benefícios para o usuário final.
* **Redação Criativa e Engajadora:** O Agente Redator gera rascunhos otimizados para o Instagram.
* **Revisão Especializada por Nicho:** O Agente Revisor garante a qualidade e a adequação do tom para diferentes públicos.
* **Flexibilidade de Assuntos:** Permite criar posts sobre diversos tópicos sem sair do programa.
* **Executável Autônomo:** Pode ser executado em qualquer computador sem a necessidade de instalação do Python.

## 🛠️ Como Utilizar

1.  **Execute o arquivo executável:** Dê um duplo clique em `main.exe` (localizado dentro da pasta `dist` após a compilação com PyInstaller) ou execute-o pelo Prompt de Comando/Terminal.
2.  **Digite o tópico:** O programa solicitará o tópico sobre o qual você deseja criar o post.
3.  **Escolha as opções:** Um menu será exibido com as seguintes opções:
    * **1. Buscar:** Executa o Agente Buscador para encontrar tendências sobre o tópico.
    * **2. Planejar post:** Executa o Agente Planejador para criar um plano com base nos lançamentos encontrados.
    * **3. Redigir rascunho:** Executa o Agente Redator para gerar um rascunho do post.
    * **4. Revisar post:** Executa o Agente Revisor para revisar o rascunho.
    * **5. Mostrar resultado final:** Exibe o resultado da última etapa concluída.
    * **6. Digitar novo assunto:** Permite inserir um novo tópico e iniciar o processo novamente.
    * **7. Sair:** Encerra o programa.
4.  **Siga as instruções:** Interaja com o programa escolhendo as opções desejadas para criar o seu post.

## ⚙️ Configuração (Para Desenvolvedores)

1.  **Clone o repositório:**
    ```bash
    git clone git@github.com:Matheuszy/agente-IA-criador-post.git
    
    ```
2.  **Crie o arquivo `.env`:**
    Adicione sua chave da API do Google Gemini no arquivo `.env`:
    ```
    GOOGLE_API_KEY=SUA_CHAVE_SECRETA
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute o script:**
    ```bash
    python main.py
    ```
5.  **Crie o executável (opcional para usuários finais):**
    ```bash
    pip install pyinstaller
    pyinstaller --onefile main.py
    ```
    O executável estará na pasta `dist`. Não se esqueça de incluir o arquivo `.env` na mesma pasta do executável para distribuição.

## 🤝 Contribuições

Contribuições são bem-vindas\! Sinta-se à vontade para abrir issues e enviar pull requests para melhorar este projeto.

## 📄 Licença

[MIT](https://opensource.org/licenses/MIT)

