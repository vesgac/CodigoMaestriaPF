import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import shap
import os
import tempfile
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
from lime import lime_tabular
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Input as KerasInput

def create_bayesian_lstm(input_shape, lstm_units=100, dropout_rate=0.7):
    model = Sequential([
        KerasInput(shape=input_shape),
        LSTM(lstm_units, activation='relu', return_sequences=False),
        Dropout(dropout_rate),  # Dropout para simular incertidumbre en la inferencia
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Cargar datos desde un archivo Excel
excel_file = "resultados_bayesian_lstm_1.xlsx"
df = pd.read_excel(excel_file)

# Cargar datos para la distribución bayesiana
bayesian_preds_load = np.load('bayesian_preds_descaled.npy')
mean_preds = bayesian_preds_load.mean(axis=0)

shap_values_descaled_loaded = np.load('shap_values_descaled_1.npy')
expected_value_descaled_loaded = np.load('expected_value_descaled_1.npy')
X_test_sample_loaded = np.load('X_test_sample.npy')
X_train = np.load('X_train.npy')
input_shape = (1, X_train.shape[1])
model_bayesian_lstm_loaded = create_bayesian_lstm(input_shape=input_shape)
model_bayesian_lstm_loaded.load_weights('model_bayesian_lstm.weights.h5')
# Cargar el escalador
with open('scaler_Y.pkl', 'rb') as file:
    scaler_Y_loaded = pickle.load(file)
print("Escalador cargado correctamente.")


# Feature names
feature_names = [
    'Direccion_lag_2', 'Direccion_lag_3', 'Humedad MN_lag_24',
    'Humedad MN_lag_23', 'Temp MN_lag_9', 'Temp MN_lag_22', 'Y_lag_1',
    'Y_lag_2', 'Y_lag_3', 'Y_lag_4', 'Y_lag_5', 'Y_lag_6', 'Y_lag_7',
    'Y_lag_8', 'Y_lag_9', 'Y_lag_10', 'Y_lag_11', 'Y_lag_12', 'Y_lag_13',
    'Y_lag_14', 'Y_lag_15', 'Y_lag_16', 'Y_lag_17', 'Y_lag_18', 'Y_lag_19',
    'Y_lag_20', 'Y_lag_21', 'Y_lag_22', 'Y_lag_23', 'Y_lag_24', 'Y_lag_25',
    'Y_lag_26', 'Y_lag_27', 'Y_lag_28', 'Y_lag_29', 'Y_lag_30', 'Y_lag_31',
    'Y_lag_32', 'Y_lag_33', 'Y_lag_34', 'Y_lag_35', 'Y_lag_36'
]
# Inicializar la app de Dash
app = dash.Dash(__name__)

# Estado inicial del índice máximo a mostrar
initial_index = 100

def _force_plot_html(base_value, shap_values, features, feature_names):
    # Generar el gráfico SHAP interactivo sin matplotlib
    force_plot = shap.force_plot(
        base_value,
        shap_values,
        features=features,
        feature_names=feature_names,
        matplotlib=False
    )

    # Construir el HTML con las bibliotecas JS necesarias
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    # Retornar el iframe con el contenido HTML
    return html.Iframe(
        srcDoc=shap_html,
        style={"width": "100%", "height": "400px", "border": "0"}
    )

def predict_descaled(x):
    """
    Realiza predicciones con el modelo cargado y las desescala usando el escalador cargado.
    """
    # Realizar predicciones con el modelo
    preds_scaled = model_bayesian_lstm_loaded.predict(x.reshape(x.shape[0], 1, x.shape[1])).flatten()
    
    # Desescalar las predicciones
    preds_descaled = scaler_Y_loaded.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    return preds_descaled

# Crear el explainer LIME
explainer = lime_tabular.LimeTabularExplainer(
    X_train,  # Conjunto de entrenamiento
    mode='regression',
    feature_names=feature_names,
    discretize_continuous=True
)

# Generar la explicación para una instancia específica
def generate_lime_plot(instance_index):
    # Seleccionar la instancia actual
    instance = X_test_sample_loaded[instance_index]

    # Generar la explicación utilizando LIME
    exp = explainer.explain_instance(
        instance,
        predict_descaled,  # Usar la función que desescala las predicciones
        num_features=10
    )

    # Obtener el HTML interactivo del gráfico LIME
    lime_html = exp.as_html()

    # Retornar un iframe con el contenido HTML
    return html.Iframe(
        srcDoc=lime_html,
        style={"width": "100%", "height": "500px", "border": "0"}
    )

# Diseñar el layout del dashboard
app.layout = html.Div(
    style={'backgroundColor': '#ccffcc', 'padding': '20px'},  # Fondo verde claro
    children=[
        html.H1("Dashboard de Predicción de Velocidad del Viento", style={'textAlign': 'center'}),
        html.Div(
            style={'display': 'flex', 'flexDirection': 'row'},  # Diseño horizontal
            children=[
                dcc.Graph(id='wind-speed-graph', style={'flex': '3'}),
                dcc.Graph(id='distribution-graph', style={'flex': '1'})
            ]
        ),
        html.Div(
            style={'textAlign': 'center', 'marginTop': '20px'},
            children=[
                html.Button("Siguiente Índice", id='next-button', n_clicks=0, style={'marginRight': '10px'}),
                html.Label("Límite Inferior: "),
                dcc.Input(id='lower-limit', type='number', placeholder='Límite Inferior', style={'marginRight': '10px'}),
                html.Label("Límite Superior: "),
                dcc.Input(id='upper-limit', type='number', placeholder='Límite Superior')
            ]
        ),
        html.Div(id='probability-display', style={'marginTop': '20px', 'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Div(
            id="shap-force-plot",
            style={"marginTop": "20px", "textAlign": "center"}
        ),
        html.Div(
            id="lime-plot",
            style={"marginTop": "20px", "textAlign": "center"}
        )
    ]
)

# Callback para actualizar las gráficas y calcular probabilidades
@app.callback(
    [Output('wind-speed-graph', 'figure'),
     Output('distribution-graph', 'figure'),
     Output('probability-display', 'children'),
     Output('shap-force-plot', 'children'),
     Output('lime-plot', 'children')],
    [Input('next-button', 'n_clicks')],
    [State('lower-limit', 'value'),
     State('upper-limit', 'value')]
)
def update_graph(n_clicks, lower_limit, upper_limit):
    # Calcular el índice máximo a mostrar basado en los clics del botón
    max_index = initial_index + n_clicks

    # Limitar el índice al rango del DataFrame
    if max_index > len(df):
        max_index = len(df)

    # Filtrar datos hasta el índice máximo actual
    filtered_df = df.iloc[:max_index]

    # Crear los índices desplazados para predicciones e intervalo de confianza
    prediction_indices = filtered_df.index + 1

    # Determinar la escala del eje Y
    y_min = min(filtered_df['Limite Inferior (99%)'].min(), filtered_df['Valores Reales (Y_test)'].min())
    y_max = max(filtered_df['Limite Superior (99%)'].max(), filtered_df['Valores Reales (Y_test)'].max())

    # Gráfica de series de tiempo
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df['Valores Reales (Y_test)'],
        mode='lines+markers',
        name='Valores Reales',
        line=dict(color='blue')
    ))
    fig1.add_trace(go.Scatter(
        x=prediction_indices,
        y=filtered_df['Predicciones (Media)'],
        mode='lines+markers',
        name='Predicciones',
        line=dict(color='orange')
    ))
    fig1.add_trace(go.Scatter(
        x=list(prediction_indices) + list(prediction_indices[::-1]),
        y=list(filtered_df['Limite Superior (99%)']) + list(filtered_df['Limite Inferior (99%)'][::-1]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',  # Rojo claro transparente
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='Intervalo de Confianza (99%)'
    ))

    # Límites para la serie de tiempo
    if lower_limit is not None:
        fig1.add_trace(go.Scatter(
            x=[0, max_index],
            y=[lower_limit, lower_limit],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Límite Inferior'
        ))
    if upper_limit is not None:
        fig1.add_trace(go.Scatter(
            x=[0, max_index],
            y=[upper_limit, upper_limit],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Límite Superior'
        ))

    fig1.update_layout(
        title="Velocidad del Viento: Valores Reales y Predicciones",
        xaxis_title="Índice",
        yaxis_title="Velocidad del Viento",
        template="simple_white",
        yaxis=dict(range=[y_min, y_max])
    )

    # Índice actual para la gráfica de distribución
    t = max_index - 1  # Índice actual

    # Gráfica de distribución bayesiana (rotada)
    histogram_data, bin_edges = np.histogram(bayesian_preds_load[:, t, 0], bins=30)
    fig2 = go.Figure()

    # Agregar el histograma (rotado)
    fig2.add_trace(go.Bar(
        y=(bin_edges[:-1] + bin_edges[1:]) / 2,  # Centros de los bins
        x=histogram_data,
        orientation='h',  # Rotación del histograma
        marker=dict(color='darkgreen', opacity=1)  # Verde oscuro sólido
    ))

    # Resaltar áreas fuera de los límites
    probabilities = []
    if lower_limit is not None:
        prob_lower = np.mean(bayesian_preds_load[:, t, 0] < lower_limit)
        probabilities.append(f"P(Vel < {lower_limit}) = {prob_lower:.2%}")
        fig2.add_trace(go.Bar(
            y=(bin_edges[:-1] + bin_edges[1:]) / 2,
            x=[count if center < lower_limit else 0 for count, center in zip(histogram_data, (bin_edges[:-1] + bin_edges[1:]) / 2)],
            orientation='h',
            marker=dict(color='red', opacity=1)  # Sólido
        ))
        fig2.add_trace(go.Scatter(
            x=[0, max(histogram_data)],
            y=[lower_limit, lower_limit],
            mode='lines',
            line=dict(color='black', dash='dash')
        ))

    if upper_limit is not None:
        prob_upper = np.mean(bayesian_preds_load[:, t, 0] > upper_limit)
        probabilities.append(f"P(Vel > {upper_limit}) = {prob_upper:.2%}")
        fig2.add_trace(go.Bar(
            y=(bin_edges[:-1] + bin_edges[1:]) / 2,
            x=[count if center > upper_limit else 0 for count, center in zip(histogram_data, (bin_edges[:-1] + bin_edges[1:]) / 2)],
            orientation='h',
            marker=dict(color='red', opacity=1)  # Sólido
        ))
        fig2.add_trace(go.Scatter(
            x=[0, max(histogram_data)],
            y=[upper_limit, upper_limit],
            mode='lines',
            line=dict(color='black', dash='dash')
        ))

    fig2.update_layout(
        title=f'Distribución de las Predicciones (Punto {t})',
        xaxis_title='Frecuencia',
        yaxis_title='Valor de la Predicción',
        template="simple_white",
        bargap=0.2,
        yaxis=dict(range=[y_min, y_max]),
        showlegend=False  # Elimina la leyenda
    )

    # Mostrar probabilidades
    probability_text = " | ".join(probabilities)
    
    shap_plot = _force_plot_html(
        base_value=np.array([expected_value_descaled_loaded]),
        shap_values=shap_values_descaled_loaded[max_index - 1, :, 0],
        features=X_test_sample_loaded[max_index - 1],
        feature_names=feature_names
    )
    
    lime_plot = generate_lime_plot(max_index - 1)

    return fig1, fig2, probability_text, shap_plot, lime_plot

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True)






