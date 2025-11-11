# Standard Imports
import streamlit as st
from textwrap import dedent
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Local Imports
from src.configs import settings

def load_data():
    """Load test data and model"""
    X_test = pd.read_parquet(settings.DATA_PROCESSED_PATH.joinpath("X_test.parquet")).assign(
        marca_da_loja=lambda _df: _df["marca_da_loja"].astype("category"),
        nome_da_loja=lambda _df: _df["nome_da_loja"].astype("category"),
    ) 
    X_train = pd.read_parquet(settings.DATA_PROCESSED_PATH.joinpath("X_train.parquet")).assign(
        marca_da_loja=lambda _df: _df["marca_da_loja"].astype("category"),
        nome_da_loja=lambda _df: _df["nome_da_loja"].astype("category"),
    ) 
    y_train = pd.read_parquet(settings.DATA_PROCESSED_PATH.joinpath("y_train.parquet"))
    y_test = pd.read_parquet(settings.DATA_PROCESSED_PATH.joinpath("y_test.parquet"))
    preprocessing_pipeline = joblib.load(settings.PROJECT_PATH.joinpath("models", "preprocessing_pipeline.pkl"))
    xgb_model = joblib.load(settings.PROJECT_PATH.joinpath("models", "xgb_optimized_model.pkl"))
    return X_test, X_train, y_train, y_test, preprocessing_pipeline, xgb_model


def get_default_sample(X_test):
    """Get a random row from X_test as default sample"""
    return X_test.sample(n=1, random_state=None).copy()


def update_datetime_features(df):
    """Update datetime-derived features from data_e_hora_do_pedido column
    
    Matches the transformation from 1.0-eda notebook:
    - dia_da_semana: dayofweek (0=Monday, 6=Sunday)
    - dia_do_mes: day of month
    - hora: hour
    - minuto: minute
    - minutos_desde_meia_noite: hour * 60 + minute
    """
    df = (
        df.
        copy()
        .assign(
            dia_da_semana=lambda _df: pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.dayofweek,
            dia_do_mes=lambda _df: pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.day,
            hora=lambda _df: pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.hour,
            minuto=lambda _df: pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.minute,
            minutos_desde_meia_noite=lambda _df: pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.hour * 60 + pd.to_datetime(_df["data_e_hora_do_pedido"]).dt.minute
        )
    )
    return df


def main():
    st.markdown(
        body=dedent(
            """
            ## Predição de Tempo de Entrega

            ### Contexto

            O modelo obteve um erro médio de aproximadamente 13 minutos nos dados de teste. As métricas são:
            """
        ),
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="MAE Treino (CV)",
            value=f"7.94 min",
            delta=f"± 0.27",
            delta_color="off",
        )
    with col2:
        st.metric(
            label="MAE Validação (CV)",
            value=f"10.01 min",
            delta=f"± 0.94",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="MAE Teste",
            value=f"13.09 min",
            delta=None,
        )
    
    # Create features info dataframe
    features_info = pd.DataFrame({
        "Feature": [
            "Marca da loja",
            "Nome da loja",
            "Data e hora do pedido",
            "Turno",
            "Serviço logístico",
            "Prioridade do pedido",
            "Distância percorrida (km)",
            "Taxa de entrega (R$)",
            "Valor total dos itens (R$)",
            "Dia da semana",
            "Dia do mês",
            "Hora do pedido",
            "Minuto do pedido",
            "Minutos desde meia-noite",
            "Pedidos na última hora"
        ],
        "Faixa/Valores": [
            "0-1 (anonimizada)",
            "0-11 (anonimizada)",
            "Datetime",
            "MANHA, ALMOCO, TARDE, JANTAR, CEIA, MADRUGADA",
            "ENTREGA_MAIS_FLEX, FULL_SERVICE, SOB_DEMANDA_ON, SOB_DEMANDA_OFF",
            "PADRAO, RAPIDA",
            "0.00 - 32.47 km",
            "R$ 4.99 - 29.99",
            "R$ 0 - 975.30",
            "0-6",
            "1-31",
            "8-22",
            "0-59",
            "520-1378",
            "0-60 pedidos"
        ]
    })
    
    st.markdown("**Variáveis utilizadas pelo modelo:**")
    st.dataframe(features_info, hide_index=True, width="stretch", height=563)
    
    st.markdown(
        body=dedent(
            """
            ---
            
            ### Playground

            Utilize a amostra abaixo para obter a predição de tempo de entrega:
            """
        ),
        unsafe_allow_html=True,
    )

    # Load data and model
    X_test, X_train, y_train, y_test, preprocessing_pipeline, xgb_model = load_data()
    
    # Initialize session state for the sample
    if "sample_df" not in st.session_state:
        st.session_state.sample_df = get_default_sample(X_test)
    
    # Reorder columns: editable first, disabled at the end
    column_order = [
        "marca_da_loja",
        "nome_da_loja",
        "data_e_hora_do_pedido",
        "turno",
        "servico_logistico",
        "prioridade_do_pedido",
        "distancia_percorrida_ate_o_cliente_km",
        "taxa_de_entrega_paga_pelo_cliente_reais",
        "valor_total_dos_itens_do_pedido_reais",
        "pedidos_ultimos_60min",
        "dia_da_semana",
        "dia_do_mes",
        "hora",
        "minuto",
        "minutos_desde_meia_noite",
    ]
    
    # Configure column settings for data_editor
    column_config = {
        "marca_da_loja": st.column_config.SelectboxColumn(
            "Marca",
            help="ID anonimizado da marca (0-1)",
            options=[0, 1],
        ),
        "nome_da_loja": st.column_config.SelectboxColumn(
            "Restaurante",
            help="ID anonimizado do restaurante (0-11)",
            options=list(range(0,12)),
        ),
        "data_e_hora_do_pedido": st.column_config.DatetimeColumn(
            "DataHora do Pedido",
            help="Data e hora do pedido",
        ),
        "turno": st.column_config.SelectboxColumn(
            "Turno",
            help="Turno do pedido",
            options=["MANHA", "ALMOCO", "TARDE", "JANTAR", "CEIA", "MADRUGADA"],
        ),
        "servico_logistico": st.column_config.SelectboxColumn(
            "Serviço Logístico",
            help="Tipo de serviço logístico",
            options=["ENTREGA_MAIS_FLEX", "FULL_SERVICE", "SOB_DEMANDA_ON", "SOB_DEMANDA_OFF"],
        ),
        "prioridade_do_pedido": st.column_config.SelectboxColumn(
            "Prioridade",
            help="Prioridade do pedido",
            options=["PADRAO", "RAPIDA"],
        ),
        "distancia_percorrida_ate_o_cliente_km": st.column_config.NumberColumn(
            "Distância (km)",
            help="Distância até o cliente (0-32.47 km)",
            min_value=0.0,
            max_value=100,
            step=0.1,
            format="%.2f",
        ),
        "taxa_de_entrega_paga_pelo_cliente_reais": st.column_config.NumberColumn(
            "Taxa de Entrega (R$)",
            help="Taxa de entrega (R$ 4.99-29.99)",
            min_value=0.0,
            max_value=50,
            step=0.01,
            format="R$ %.2f",
        ),
        "valor_total_dos_itens_do_pedido_reais": st.column_config.NumberColumn(
            "Valor Total (R$)",
            help="Valor total dos itens (R$ 0-975.30)",
            min_value=0.0,
            max_value=1000.0,
            step=0.10,
            format="R$ %.2f",
        ),
        "dia_da_semana": st.column_config.NumberColumn(
            "Dia da Semana",
            help="Dia da semana (0=Segunda, 6=Domingo) - Calculado automaticamente",
            disabled=True,
        ),
        "dia_do_mes": st.column_config.NumberColumn(
            "Dia do Mês",
            help="Dia do mês (1-31) - Calculado automaticamente",
            disabled=True,
        ),
        "hora": st.column_config.NumberColumn(
            "Hora",
            help="Hora do pedido (8-22) - Calculado automaticamente",
            disabled=True,
        ),
        "minuto": st.column_config.NumberColumn(
            "Minuto",
            help="Minuto do pedido (0-59) - Calculado automaticamente",
            disabled=True,
        ),
        "minutos_desde_meia_noite": st.column_config.NumberColumn(
            "Minutos desde Meia-Noite",
            help="Minutos desde meia-noite (520-1378) - Calculado automaticamente",
            disabled=True,
        ),
        "pedidos_ultimos_60min": st.column_config.NumberColumn(
            "Pedidos (últimos 60min)",
            help="Número de pedidos na última hora (0-60)",
            min_value=0,
            max_value=100,
            step=1,
        ),
    }
    
    # Display editable dataframe
    df_edited = st.data_editor(
        st.session_state.sample_df,
        column_config=column_config,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_order=column_order,
    )
    
    # Update datetime-derived features if datetime changed
    if not df_edited["data_e_hora_do_pedido"].equals(st.session_state.sample_df["data_e_hora_do_pedido"]):
        df_edited = update_datetime_features(df_edited)
    
    # Update session state with edited values
    st.session_state.sample_df = df_edited
    
    # Buttons in columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Resetar Amostra", use_container_width=True):
            st.session_state.sample_df = get_default_sample(X_test)
    
    with col2:
        predict_button = st.button("Prever Tempo de Entrega", use_container_width=True, type="primary")
    
    # Prediction section
    if predict_button:
        with st.spinner("Realizando predição..."):
            prediction = xgb_model.predict(df_edited)[0]
            st.session_state.last_prediction = prediction
    
    # Display prediction persistently
    if "last_prediction" in st.session_state:
        st.markdown(f"#### Tempo de Entrega Previsto: **{st.session_state.last_prediction:.2f} minutos**")
    
    st.markdown("---")
    
    # Advanced section: Custom model training
    st.markdown(
        body=dedent(
            """
            ### Personalização de Parâmetros

            Configure os hiperparâmetros do XGBoost e treine um novo modelo com os dados de treinamento.
            """
        ),
        unsafe_allow_html=True,
    )
    
    with st.expander("Configurar Hiperparâmetros do XGBoost", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider(
                "n_estimators",
                min_value=100,
                max_value=2000,
                value=2000,
                step=100,
                help="Número de árvores no ensemble"
            )
            
            learning_rate = st.slider(
                "learning_rate",
                min_value=0.001,
                max_value=0.1,
                value=0.0055,
                help="Taxa de aprendizado"
            )
            
            max_depth = st.slider(
                "max_depth",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Profundidade máxima de cada árvore"
            )
        
        with col2:
            subsample = st.slider(
                "subsample",
                min_value=0.05,
                max_value=1.0,
                value=0.205,
                step=0.05,
                help="Fração de amostras usadas por árvore"
            )
            
            colsample_bytree = st.slider(
                "colsample_bytree",
                min_value=0.05,
                max_value=1.0,
                value=0.727,
                step=0.05,
                help="Fração de features usadas por árvore"
            )
        
        with col3:
            min_child_weight = st.slider(
                "min_child_weight",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Peso mínimo necessário em um nó filho"
            )
            
            gamma = st.slider(
                "gamma",
                min_value=0.0,
                max_value=5.0,
                value=0.553,
                step=0.1,
                help="Redução mínima de perda para criar nova partição"
            )
        
        if st.button("Treinar Modelo Personalizado", use_container_width=True):
            st.session_state._just_trained = True
            with st.spinner("Treinando modelo com validação cruzada..."):
                # Create custom XGBoost model with user parameters
                custom_xgb = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                )
                
                # Create pipeline with preprocessing + custom model
                custom_pipeline = Pipeline(
                    preprocessing_pipeline.steps + [("xgboost_custom", custom_xgb)]
                )
                
                # Cross-validation on training data
                kfold = KFold(n_splits=3, shuffle=False)

                cv_results = cross_validate(
                    estimator=custom_pipeline,
                    X=X_train,
                    y=y_train,
                    scoring="neg_mean_absolute_error",
                    cv=kfold,
                    return_train_score=True,
                )
                
                # Calculate MAE
                mae_train = -cv_results["train_score"]
                mae_val = -cv_results["test_score"]
                
                # Train final model on all training data
                custom_pipeline.fit(X_train, y_train)
                
                # Calculate MAE on test set
                y_pred_test = custom_pipeline.predict(X_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                
                # Store custom model in session state
                st.session_state.custom_model = custom_pipeline
                st.session_state.mae_train = mae_train
                st.session_state.mae_val = mae_val
                st.session_state.mae_test = mae_test
                
                # Display results
                st.success("✅ Modelo treinado com sucesso!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="MAE Treino (CV)",
                        value=f"{mae_train.mean():.2f} min",
                        delta=f"± {mae_train.std():.2f}",
                        delta_color="off",
                    )
                with col2:
                    st.metric(
                        label="MAE Validação (CV)",
                        value=f"{mae_val.mean():.2f} min",
                        delta=f"± {mae_val.std():.2f}",
                        delta_color="off",
                    )
                with col3:
                    st.metric(
                        label="MAE Teste",
                        value=f"{mae_test:.2f} min",
                        delta_color="off",
                    )
    
    # Display metrics if custom model exists (persist across reruns) - OUTSIDE expander
    if "custom_model" in st.session_state and "mae_train" in st.session_state:
        if not st.session_state.get('_just_trained', False):
            st.success("✅ Modelo personalizado disponível!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="MAE Treino (CV)",
                    value=f"{st.session_state.mae_train.mean():.2f} min",
                    delta=f"± {st.session_state.mae_train.std():.2f}",
                    delta_color="off",
                )
            with col2:
                st.metric(
                    label="MAE Validação (CV)",
                    value=f"{st.session_state.mae_val.mean():.2f} min",
                    delta=f"± {st.session_state.mae_val.std():.2f}",
                    delta_color="off",
                )
            with col3:
                st.metric(
                    label="MAE Teste",
                    value=f"{st.session_state.mae_test:.2f} min",
                    delta_color="off",
                )
        else:
            # Reset the flag after showing metrics once
            st.session_state._just_trained = False
    
    # If custom model exists in session state, allow predictions
    if "custom_model" in st.session_state:
        st.markdown("---")
        if st.button("Prever com Modelo Personalizado", use_container_width=True):
            custom_prediction = st.session_state.custom_model.predict(df_edited)[0]
            st.session_state.last_custom_prediction = custom_prediction
        
        # Display custom prediction persistently
        if "last_custom_prediction" in st.session_state:
            st.markdown(f"#### Tempo de Entrega Previsto (Modelo Personalizado): **{st.session_state.last_custom_prediction:.2f} minutos**")


if __name__ == "__main__":
    main()