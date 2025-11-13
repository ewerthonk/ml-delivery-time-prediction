# Standard Imports
import streamlit as st
from textwrap import dedent

# Local Imports


def main():
    st.markdown(
        body=dedent(
            """
            ## Feature Engineering

            ### Contexto
            
            Adicionar features:
            1. Número de pedidos na loja na última hora
            2. Precipitação na última hora.

            ### Execução
            
            Uilizar `OpenWeatherAPI` e transformações `pandas`. A operação com OpenWeatherAPI ficou inviável devido ao custo da API e a distribuição geográfica das lojas (50 R$ por cidade para obter dados históricos) e foi desconsiderada.

            ### Considerações

            A feature de `Número de pedidos na loja na última hora` foi adicionada, mas teve pouco impacto em termos de correlação e explicabilidade da variação do target.

            ## Data Preprocessing

            ### Contexto

            1. Desconsiderar coluna `data_e_hora_do_pedido`
            2. Utilizar encoding nas variáveis categóricas
            3. Realizar splitting

            ### Execução

            - Coluna `data_e_hora_do_pedido` removida com DropFeatures da biblioteca *feature-engine*.
            - Encoders aplicados nas variáveis categóricas:
                - `turno`: Ordinal Encoder com ordem ["MANHA", "ALMOCO", "TARDE", "JANTAR", "CEIA", "MADRUGADA"], 
                representando a progressão natural dos períodos do dia.
                - `prioridade_do_pedido`: Ordinal Encoder com ordem ["PADRAO", "RAPIDA"], 
                capturando a hierarquia de urgência dos pedidos.
                - `marca_da_loja` e `nome_da_loja`: Count Frequency Encoder, 
                transformando categorias em suas frequências absolutas no dataset.
                - `servico_logistico`: One-Hot Encoder.
            - Split: 80/20 para Treino/Teste e Validação Cruzada com 3 folds.

            ### Considerações

            Dados prontos para modelagem de Machine Learning.
            """
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
