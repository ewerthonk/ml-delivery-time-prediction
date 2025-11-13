# Standard Imports
import streamlit as st
from textwrap import dedent


def main():
    st.markdown(
        body=dedent(
            """
            ## Entendimento do Negócio
            
            ### Contexto

            O objetivo do projeto é desenvolver um modelo preditivo para estimar o tempo de entrega de pedidos em uma rede de restaurantes de delivery, utilizando dados históricos de pedidos.
            
            1. Planilha inicial contém 29 colunas e é extraída do iFood com limitação de 90 dias de dados, contemplando o período 09/08/2025 a 07/11/2025. 
            2. A descrição das colunas levantado com o time de negócio e anotada no arquivo `raw_data_dict.json`.
            3. Os dados de marca da loja e nome da loja devem ser anonimizados por privacidade.

            ### Regras de negócio

            1. Há 8 colunas - relacionadas a tempos parciais da entrega - que não estão disponíveis no momento do pedido e devem ser desconsiderados.
            2. A coluna `STATUS FINAL` contém pedidos cancelados que devem ser desconsiderados.
            3. Aconselhou-se a desconsiderar a coluna `TEMPO PROMETIDO DE ENTREGA (MIN)` para a aplicação prática do modelo - já que as predições podem ser usadas para definir esse tempo prometido.
            4. As colunas `TEMPO DE ATRASO EM RELAÇÃO AO TEMPO PROMETIDO DE ENTREGA (MIN)` e `FRETE COBRADO DO RESTAURANTE (APENAS SOB DEMANDA)` também podem ser desconsiderados.
            5. Os fatores **meteorológicos**, **demanda da cozinha**, **disponibilidade de entregadores** e **alimentos quentes/frios** são relevantes para o tempo de entrega, mas não estão presentes na base de dados.

            ### Solução Atual
            
            Atualmente, o negócio utiliza a média histórica dos tempos de entrega para definir a faixa de tempo prometido ao cliente.

            ### Classificação do Problema

            Regressão com Aprendizado de Máquina Supervisionado.

            ### Exemplo do json: `raw_data_dict.json`

            ```json
            {
                "nome_do_campo_antes_limpeza": "MARCA DA LOJA",
                "nome_do_campo_apos_limpeza": "marca_da_loja",
                "tipo": "category",
                "descrição": "Identificador codificado da marca da loja (anonimizado por privacidade)",
                "disponivel_na_hora_do_pedido": true
            }
            ```
            """
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
