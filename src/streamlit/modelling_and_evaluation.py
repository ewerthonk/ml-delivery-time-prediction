# Standard Imports
import streamlit as st
import pandas as pd
from textwrap import dedent

# Local Imports
from src.configs import settings

def main():
    st.markdown(
        body=dedent(
            f"""
            ## Modelling

            ### Contexto
            
            Utilizar um modelo baseline, um modelo de XGBoost inicial e um modelo de XGBoost otimizado com hyperparameter tuning. A escolha pelo modelo XGBoost foi arbitrária.

            ### Execução
            
            Modelos:
            1. Modelo baseline: Dummy Regressor com estratégia "mean".
            2. Modelo XGBoost inicial com n_estimators = 100.
            3. Modelo XGBoost otimizado com hyperparameter tuning usando `Optuna`.

            Espaço de busca para hyperparameter tuning:
            - **`n_estimators`**: 100 a 2000 (passo de 100) - Número de árvores no ensemble
            - **`learning_rate`**: 0.001 a 0.1 (escala logarítmica) - Taxa de aprendizado
            - **`max_depth`**: 1 a 10 - Profundidade máxima de cada árvore
            - **`subsample`**: 0.05 a 1.0 - Fração de amostras usadas por árvore (bagging)
            - **`colsample_bytree`**: 0.05 a 1.0 - Fração de features usadas por árvore
            - **`min_child_weight`**: 1 a 20 - Peso mínimo necessário em um nó filho
            - **`gamma`**: 0 a 5.0 - Redução mínima de perda para criar nova partição

            ### Conclusões

            Modelos atendem aos critérios de avaliação estabelecidos.

            ## Evaluation

            ### Contexto

            As duas métricas observadas foram MAE (primária) e RMSE (secundária) com o objetivo de compreender o desempenho do modelo e detectar Underfit e Overfit.

            ### Execução

            - As métricas foram anotadas no conjunto de Treino/Test numa validação cruzada de 3 folds.
            - Utilizou-se da **curva de aprendizado** e do **gráfico de resíduos** dos modelos para analisar se o número de observações foi suficiente e detectar overfit e underfit.

            ### Considerações

            Os resultados dos modelos em termos de métricas (MAE e RMSE) de treino, validação e teste são: 
            """
        ),
        unsafe_allow_html=True,
    )
    
    st.dataframe(
        data=pd.read_parquet(path=settings.DATA_PROCESSED_PATH.joinpath("results.parquet")),
        hide_index=True,
    )

    st.markdown(
        body=dedent(
            f"""
            ### Baseline

            O Dummy Regressor (que prediz o valor da média dos dados de treino - aproximadamente 35,4 minutos) apresentou MAE de aproximadamente 12 minutos em treino e validação, aumentando para 14.41 minutos no teste. 
            Este modelo serve como referência mínima de desempenho, mostrando que qualquer modelo preditivo deve superar estes valores para ser considerado útil.
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_dummy.png"))
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_dummy.png"))
    st.markdown(
        body=dedent(
            f"""
            Das visualizações, observa-se o R2 de treino igual a 0 e o R2 de teste é negativo, indicando que o modelo não explica a variabilidade dos dados. Não há observações relevantes para o gráfico de curva de aprendizado, já que a média é utilizada.
            
            ---
            """
        ),
        unsafe_allow_html=True,
    )
    

    st.markdown(
        body=dedent(
            f"""
            ### XGBoost Inicial

            O modelo XGBoost inicial (com n_estimators = 100) apresentou MAE de aproximadamente 11 minutos em treino, aumentando para 13.5 minutos no teste. 
            Este modelo serve como uma melhoria em relação ao Dummy Regressor, mas com pouco ganho de desempenho em relação à média (modelo anterior).
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_xgb_initial.png"))
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_xgb_initial.png"))
    st.markdown(
        body=dedent(
            f"""
            Das visualizações, observa-se um claro **overfit**, caracterizado pela diferença entre as curvas de aprendizado e os parâmetros. Além disso, nota-se que um aumento do número de observações ainda pode ser benéfico para o modelo.

            ---
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        body=dedent(
            f"""
            ### XGBoost otimizado

            Parâmetros definidos pela otimização:

            ```json
            {{
                "n_estimators": 2000,
                "learning_rate": 0.0055127883672954235,
                "max_depth": 5,
                "subsample": 0.20501390052078072,
                "colsample_bytree": 0.7265258604534922,
                "min_child_weight": 1,
                "gamma": 0.5532158798897284
            }}
            ```

            O modelo XGBoost otimizado apresentou MAE de aproximadamente 10 minutos em treino, aumentando para 13 minutos no teste.
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_xgb_optimized.png"))
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_xgb_optimized.png"))
    st.markdown(
        body=dedent(
            f"""
            Das visualizações, observa-se uma redução do **overfit**, caracterizado pela aproximação das curvas de aprendizado e dos parâmetros e melhor desempenho geral.
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        body=dedent(
            f"""
            ## Considerações Finais
            
            1. Não há ganhos significativos da abordagem de machine learning em relação ao baseline com a média (35,4 minutos) com erro médio de 14,4 minutos para 13 minutos. 
            2. Há indícios de que as features do dataset não representem o problema real de forma satisfatória, especialmente considerando os fatores levantados durante a fase de Business Understanding (meteorológicos, demanda da cozinha, disponibilidade de entregadores e pedidos com alimentos quentes/frios) e as considerações da fase de Data Understanding.
            3. O ganho massivo de desempenho com machine learning ocorre em até 6000 observações, fortalecendo a consideração #1. Apesar disso, o modelo não atingiu um plateau em termos de aprendizado.
            4. Modelar o problema de forma isolada para cada restaurante pode trazer ganhos e justificar a implementação do modelo, especialmente considerando a curva ABC de restaurantes por volume de pedidos e criticidade de demanda.
            5. Há indícios de que o comportamento da distribuição do tempo de entrega varia ao longo do tempo, considerando o formato do split de dados e a diferença da média dos dados de treino (35,4 minutos) e a média geral (37 minutos).
            """
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()