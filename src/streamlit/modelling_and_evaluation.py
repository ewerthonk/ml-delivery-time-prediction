# Standard Imports
import streamlit as st
import pandas as pd
from textwrap import dedent

# Local Imports
from src.configs import settings


def main():
    st.markdown(
        body=dedent(
            """
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
            - **`alpha`**: 0 a 10.0 - Regularização L1
            - **`lambda`**: 1 a 10.0 - Regularização L2

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
        data=pd.read_parquet(
            path=settings.DATA_PROCESSED_PATH.joinpath("results.parquet")
        ),
        hide_index=True,
    )

    st.markdown(
        body=dedent(
            """
            ### Baseline

            O Dummy Regressor (que prediz o valor da média dos dados de treino - aproximadamente 37 minutos) apresentou MAE de aproximadamente 12,9 minutos em treino, validação e teste.
            Este modelo serve como referência mínima de desempenho.
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_dummy.png"))
    st.image(image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_dummy.png"))
    st.markdown(
        body=dedent(
            """
            Das visualizações, observa-se o R2 de treino e teste igual a 0, indicando que o modelo não explica a variabilidade dos dados. Não há observações relevantes para o gráfico de curva de aprendizado, já que a média é utilizada.
            
            ---
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        body=dedent(
            """
            ### XGBoost Inicial

            O modelo XGBoost inicial (com n_estimators = 100) apresentou MAE de aproximadamente 5,7 minutos em treino, aumentando para 9,8 minutos no teste. 
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(
        image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_xgb_initial.png")
    )
    st.image(
        image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_xgb_initial.png")
    )
    st.markdown(
        body=dedent(
            """
            Das visualizações, observa-se um comportamento típico de **overfit**, caracterizado pela diferença entre as curvas de aprendizado e os parâmetros. Além disso, nota-se que um aumento do número de observações ainda pode ser benéfico para o modelo.

            ---
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        body=dedent(
            """
            ### XGBoost otimizado

            Parâmetros definidos pela otimização:

            ```json
            {
                "n_estimators": 785,
                "learning_rate": 0.012,
                "max_depth": 8,
                "subsample": 0.82,
                "colsample_bytree": 0.76,
                "min_child_weight": 18,
                "gamma": 4.42,
                "alpha": 2.38,
                "lambda": 9.82,
            }
            ```

            O modelo XGBoost otimizado apresentou MAE de aproximadamente 7,4 minutos em treino, aumentando para 9,5 minutos no teste.
            """
        ),
        unsafe_allow_html=True,
    )
    st.image(
        image=settings.RESOURCES_PATH.joinpath("visualizations", "lc_xgb_optimized.png")
    )
    st.image(
        image=settings.RESOURCES_PATH.joinpath("visualizations", "rp_xgb_optimized.png")
    )
    st.markdown(
        body=dedent(
            """
            Das visualizações, observa-se uma redução do **overfit**, caracterizado pela aproximação das curvas de aprendizado e dos parâmetros e melhor desempenho geral.
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        body=dedent(
            """
            ## Considerações Finais

            1. O ganho de desempenho da abordagem de machine learning em relação ao baseline foi a redução do erro médio (MAE) de 12,8 minutos para 9,5 minutos (redução de 25,8%). Considerando o tempo médio de entrega de aproximadamente 37 minutos, a viabilidade da implementação depende da estratégia do negócio.
            2. Há indícios de que as features do dataset podem não representar o problema real de forma satisfatória, especialmente considerando os fatores levantados durante a fase de Business Understanding (meteorológicos, demanda da cozinha, disponibilidade de entregadores e pedidos com alimentos quentes/frios) e as considerações da fase de Data Understanding (correlações baixas).
            3. O ganho massivo de desempenho com machine learning ocorre em até 6000 observações, indicando que um maior número de observações pode gerar melhores resultados. Apesar disso, o modelo não atingiu um plateau em termos de aprendizado.
            4. Modelar o problema de forma isolada para cada restaurante pode trazer ganhos e justificar a implementação do modelo, especialmente considerando a curva ABC de restaurantes por volume de pedidos e criticidade de demanda. Essa abordagem, aliada à adição das features citadas no tópico 2, podem justificar a evolução de esforços no projeto.
            """
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
