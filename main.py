from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session  # Importamos Session
from google.adk.tools import google_search
from google.genai import types
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")

# Criar o serviço de sessão uma vez
session_service = InMemorySessionService()

def call_agent(agent: Agent, message_text: str, user_id: str = "user_sistema", session_id: str = "session_unica"):
    session = session_service.create_session(app_name=agent.name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])
    final_response = ""
    for event in runner.run(user_id=user_id, session_id=session.id, new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text
                    final_response += "\n"
    return final_response

def criar_agente(name, model, description, instruction, tools=None):
    if tools is None:
        tools = []
    agente = Agent(
        name=name,
        model=model,
        description=description,
        tools=tools,
        instruction=instruction
    )
    return agente

# --- Definição dos Agentes (mantendo como estão por enquanto) ---
def agente_buscador(topico, data_de_hoje):
    buscador = criar_agente(
        name="agente_buscador",
        model="gemini-2.0-flash",
        description="Agente de busca no google",
        tools=[google_search],
        instruction="""
        você é um especilista em buscar tendências de post e mercado relacionado a tecnologia. Você foi contratado pela Codex System, uma startup especilizada em criar soluções em python para marketing digital, jurídico, profissionais da saúde, varejo, restaurantes/lanchonetes, logística, firmas de contabilidade ou finanças e imobiliárias. Seu trabalho é ser um assistente atendo com as novidades que você pode pesquisar no google search(google search) e saber quais soluções essas públicos precisam que sejam criadas, focando nas dores reais e cases reais e muito impactantes, com grande nomes. Você deve sempre se basear na atualidade, no máximo até 5 meses para selecionar o tópico
        """
    )
    entrada_do_agente_buscador = f"Tópico: {topico}\nData de hoje: {data_de_hoje}"
    lancamentos_buscados = call_agent(buscador, entrada_do_agente_buscador)
    return lancamentos_buscados

def agente_planejador(topico, lancamentos_buscados):
    planejador = criar_agente(
        name="agente_planejador",
        model="gemini-2.0-flash",
        instruction="""
        Você é um planejador de posts, especialista em redes sociais. Com base na lista na pesquisa do agente buscador. Você deve:
        usar a ferramentas do google search (google search) para criar um plano sobre quais os pontos mais relevantes que
        poderíamos abordar em um post
        sobre cada um deles. Você pode usar o google search para achar mais informações sobre os temas. Ao planejar os posts, você deve sempre focar nas vantagens que as soluções abordadas anteriormente trazem para o nicho escolhido, e sempre focar nisso. Nunca em como vai ser construido, isso não interessa ao consumir final

        ao final você irá escolher o tema mais relevante entre suas pesquisas e retornar esse tema, seus pontos mais relevantes
        e um plano com os assuntos a serem abordados que será
        escrito posteriormente
        """,
        description="Agente que planeja posts",
        tools=[google_search]
    )
    entrada_do_agente_planejador = f"Tópico:{topico}\nLançamentos buscados: {lancamentos_buscados}"
    plano_do_post = call_agent(planejador, entrada_do_agente_planejador)
    return plano_do_post

def agente_redator(topico, plano_de_post):
    redator = criar_agente(
        name="agente_redator",
        model="gemini-2.5-pro-preview-03-25",
        instruction="""
        Você é um Redator Criativo especializado em criar posts virais para redes sociais.
        Você escreve posts para a startuo Codex System, uma pequena empreitada de dois amigos apaixonados por solucionar problemas em programação python
        Utilize o tema fornecido no plano de post e os pontos mais relevantes fornecidos e, com base nisso,
        escreva um rascunho de post para Instagram sobre o tema indicado.
        O post deve ser engajador, informativo, com linguagem simples e incluir 2 a 4 hashtags no final.
        """,
        description="Agente redator de posts engajadores para Instagram"
    )
    entrada_do_agente_redator = f"Tópico: {topico}\nPlano de post: {plano_de_post}"
    rascunho = call_agent(redator, entrada_do_agente_redator)
    return rascunho

def agente_revisor(topico, rascunho_gerado):
    revisor = criar_agente(
        name="agente_revisor",
        model="gemini-2.5-pro-preview-03-25",
        instruction="""
        Você é um Editor e Revisor de Conteúdo meticuloso, especializado em posts para redes sociais, com foco no Instagram.
        Por ter um público marketing digital, jurídico, profissionais da saúde, varejo, restaurantes/lanchonetes, logística, firmas de contabilidade ou finanças e imobiliárias, use um tom de escrita adequado, que tenha palavras chave do público escolhido.
        Revise o rascunho de post de Instagram abaixo sobre o tópico indicado, verificando clareza, concisão, correção e tom.
        Se o rascunho estiver bom, responda apenas 'O rascunho está ótimo e pronto para publicar!'.
        Caso haja problemas, aponte-os e sugira melhorias.
        """,
        description="Agente revisor de post para redes sociais."
    )
    entrada_do_agente_revisor = f"Tópico: {topico}\nRascunho: {rascunho_gerado}"
    texto_revisado = call_agent(revisor, entrada_do_agente_revisor)
    return texto_revisado

def main():
    data_de_hoje = date.today().strftime("%d/%m/%Y")
    print("🚀 Iniciando o Sistema de Criação de Posts para Instagram com 4 Agentes 🚀")

    topico = input("❓ Por favor, digite o TÓPICO sobre o qual você quer criar o post de tendências: ")

    if not topico:
        print("Você esqueceu de digitar o tópico.")
        return

    print(f"Maravilha! Vamos trabalhar no post sobre novidades em {topico}.")

    lancamentos = None
    plano_de_post = None
    rascunho_de_post = None
    texto_revisado = None

    while True:
        print("\nOpções:")
        print("1. Buscar")
        print("2. Planejar post")
        print("3. Redigir rascunho")
        print("4. Revisar post")
        print("5. Mostrar resultado final")
        print("6. Digitar novo assunto")
        print("7. Sair") # Alterei a opção de sair para 7

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            lancamentos = agente_buscador(topico, data_de_hoje)
            print("\nLançamentos encontrados:")
            print(lancamentos)
        elif opcao == "2":
            if lancamentos:
                plano_de_post = agente_planejador(topico, lancamentos)
                print("\nPlano do post:")
                print(plano_de_post)
            else:
                print("Por favor, busque os lançamentos primeiro (opção 1).")
        elif opcao == "3":
            if plano_de_post:
                rascunho_de_post = agente_redator(topico, plano_de_post)
                print("\nRascunho do post:")
                print(rascunho_de_post)
            else:
                print("Por favor, planeje o post primeiro (opção 2).")
        elif opcao == "4":
            if rascunho_de_post:
                texto_revisado = agente_revisor(topico, rascunho_de_post)
                print("\nTexto revisado:")
                print(texto_revisado)
            else:
                print("Por favor, redija o rascunho primeiro (opção 3).")
        elif opcao == "5":
            if texto_revisado:
                print("\nResultado Final:")
                print(texto_revisado)
            elif rascunho_de_post:
                print("\nResultado Final (Rascunho não revisado):")
                print(rascunho_de_post)
            elif plano_de_post:
                print("\nResultado Final (Apenas plano):")
                print(plano_de_post)
            elif lancamentos:
                print("\nResultado Final (Apenas lançamentos):")
                print(lancamentos)
            else:
                print("Nenhum resultado gerado ainda.")
        elif opcao == "6":
            print("\nVamos começar com um novo assunto.")
            topico = input("❓ Digite o novo TÓPICO: ")
            print(f"Maravilha! Vamos trabalhar no post sobre novidades em {topico}.")
            # Limpar as variáveis de estado para o novo tópico
            lancamentos = None
            plano_de_post = None
            rascunho_de_post = None
            texto_revisado = None
        elif opcao == "7": # A opção de sair agora é 7
            print("Encerrando o sistema.")
            break
        else:
            print("Opção inválida. Por favor, escolha uma das opções.")

if __name__ == "__main__":
    main()