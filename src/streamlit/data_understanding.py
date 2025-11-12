# Standard Imports
import streamlit as st
import pandas as pd
from textwrap import dedent
from pathlib import Path

# Local Imports
from src.configs import settings

def main():
    st.markdown(
        body=dedent(
            f"""
            ## Exploratory Data Analysis (EDA)
            
            ### EDA: Qualidade dos Dados

            #### Contexto

            **Objetivo**: descobrir se os dados se comportam como deveriam se comportar em termos de nomes das colunas, tamanho do dataset, tipos de dados, valores ausentes e duplicados.
            **Método**: utilizar `ydata_profiling`.

            #### Execução
            
            1. **Desafio**: planilhas com dados codificados em latin-1.<br>**Solução**: usar unidecode na ingestão.
            2. **Desafio**: colunas com nomes longos, acentos e espaços.<br>**Solução**: renomear colunas para nomes curtos e sem espaços.

            #### Considerações

            1. 14 colunas (13 features + 1 target). 19.285 pedidos.
            2. 5 features categóricas, 6 numéricas, 1 ID único, 1 datetime, e 1 target numérico.
            3. Valores ausentes removidos com filtros de negócio.

            ### EDA: Entendimento dos Dados

            #### Contexto

            **Objetivo**: descobrir as características e relacionamento dos dados em termos de medidas de localização, variabilidade e distribuição.
            
            **Método**: utilizar `ydata_profiling`.

            #### Execução

            1. **Desafio**: `status_final_do_pedido` passa a ficar constante após filtragem.<br>**Solução**: desconsiderar.
            2. **Desafio**: `id_da_loja`, `id_curto_do_pedido` e `id_completo_do_pedido` não são úteis para a predição.<br>**Solução**: desconsiderar.
            3. **Desafio**: gerar variáveis temporais a partir de `data_e_hora_do_pedido`.<br>**Solução**: utilizar `feature-engine`.

            #### Considerações

            ##### Qualidade 

            Usar ordinal encoding para `turno`, `prioridade_do_pedido`, frequency encoding para `marca_da_loja`, `nome_da_loja` e one-hot encoding para `servico_logistico`.

            ##### Entendimento

            **Correlações**
            
            Nada muito animador. A feature com maior correlação é `distancia_percorrida_ate_o_cliente_km` com 0,4 seguida por `nome_da_loja` com 0,3. Apesar disso, as relações lineares nos gráficos de dispersão são aparentes.
            
            **Medidas Estatísticas**

            1. O target tem uma distribuição aproximadamente normal, com média de 37 minutos e desvio padrão de 18 minutos.
            2. A única variável altamente desbalanceada é `prioridade_do_pedido`, onde quase 80% dos dados são de prioridade padrão.
            3. A distância até o cliente e valor do pedido tem distribuições com boa variabilidade e cauda longa com assimetria positiva (para a esquerda).

            **Poder Preditivo**
            
            Baixo. Somente duas features iniciais apresentaram algum poder preditivo: a taxa de entrega paga pelo cliente (4,1%) e o serviço logístico (0,3%).

            **Visualização de Árvore de Decisão**

            Adiciona às conclusões anteriores. Com destaque para a distância até o cliente, loja e features temporais.
            
            #### Gerais

            1. Há baixa correlação e poder preditivo geral das features o target.
            2. Vale a pena gerar novas features baseadas nas regras de negócio. Contudo, a única variável disponível para engenharia de feature é a de demanda nos últimos 60 minutos.
            3. As features temporais não apresentaram tanta relevância com o target como esperado. É provável que uma única feature que represente a sazonalidade intradia - que também captura o efeito do trânsito - seja suficiente.
            4. A feature referente à loja captura parte do efeito de tamanho da cidade e de seu próprio funcionamento operacional.
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Relatório", "Poder Preditivo", "Árvore de Decisão"])
    
    with tab1:
        with open(settings.RESOURCES_PATH.joinpath("reports", "deliveries_data_profile_report.html"), "r", encoding="utf-8") as html_file:
            html_content = html_file.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    
    with tab2:
        df_pps = pd.read_parquet(settings.DATA_PROCESSED_PATH.joinpath("pps_predictors.parquet"), engine="pyarrow")
        
        st.dataframe(
            data=df_pps,
            width="stretch",
            hide_index=True,
            column_config={
                "x": st.column_config.TextColumn("Feature"),
                "ppscore": st.column_config.NumberColumn(
                    "PPS Score",
                    format="%.3f"
                ),
                "baseline_score": st.column_config.NumberColumn(
                    "Baseline Score",
                    format="%.3f"
                ),
                "model_score": st.column_config.NumberColumn(
                    "Model Score",
                    format="%.3f"
                )
            }
        )
    
    with tab3:
        svg_path = settings.RESOURCES_PATH.joinpath("visualizations", "tree_regressor.svg")
        with open(svg_path, "r", encoding="utf-8") as svg_file:
            svg_content = svg_file.read()
        st.image(svg_content, width="stretch")

if __name__ == "__main__":
    main()