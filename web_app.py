import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout='wide')
st.markdown(f""" 
        <div style='text-align:center;'>
            <h1>Estimativa da trajetória - Manipulador Paralelo 3RRR</h1>
        </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["O manipulador",
                                  "Modelo Simple RNN",
                                  "Modelo LSTM",
                                  "Modelo GRU",
                                  "Modelo CNN",
                                  "Comparação dos modelos"])

with tab1:
    col1, col2 = st.columns(2)
    col1.header("Objetivos gerais")
    col1.write("Esse é um Manipulador Robótico de Cadeia Cinemática Paralela (PKM)\
               presente no [Laboratório de Dinâmica da EESC-USP](https://dinamica.eesc.usp.br/index.html).")
    col1.write('O objetivo é, utilizando dados provenientes dos sensores instalados\
               na estrutura do manipulador (elos) e motores, implementar modelos de\
               Machine Learning para realizar uma previsão da movimentação em \
               cada coordenada de interesse.')
    col1.write("Essas coordenadas de interesse são obtidas por uma câmera \
               instalada acima do manipulador, que captura o ponto \
               preto (figura ao lado) que diz respeito as coordenadas X, Y e\
               orientação alpha do efetudador do manipulador.")
    col1.write("A câmera então, retorna essas coordenadas ao longo do tempo em \
               que o manipulador esteve em movimentação. Nesse caso em específico,\
               foi dado um comando de movimentação em apenas um motor rotativo\
               para que o robô se movimentasse durante 10 segundos e por fim\
               se estabilizasse.")
    col1.write("Os dados que foram utilizados como 'features' (ou recursos)\
               para o treinamento dos diferentes modelos de Machine Learning são:\
               ")
    col1.write('- Posição angular proveniente do primeiro, segundo e terceiro\
                motor, capturado por encoders (enc1, enc2, enc3).')
    col1.write('- Deformação de cada elo das cadeias cinemáticas, capturadas\
               por extensômetros (strain1, strain2, strain3, strain4, strain5, strain6).')
    col1.write("Esses dados coletados estão dispostos logo abaixo para visualização \
               em formato de tabelas e gráficos.")
    col1.write("OBS: Alguns desses dados já foram previamente tratados para serem\
               apresentados aqui, visto que após a coleta, alguns deles apresentaram\
               distorções devido a ruídos de alta frequência e alta amplitude\
               proveniente de máquinas elétricas instaladas na bancada do manipulador.")
    col2.image("imgs/3RRR_flex.jpg",use_column_width=True)
    st.header("Dados do manipulador")
    col1, col2, col3 = st.columns([1,2.5,2.5])
    df = pd.read_csv("dados/camera_strain_traj1_data_trat.csv")
    colunas = df.columns
    colunas = colunas.drop('tempo')
    select = col1.selectbox("Escolha qual dado gostaria de visualizar ao longo do tempo",colunas)
    x = df['tempo']
    if select == "enc1":
        y = df[select]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines'))
        fig1.update_layout(title="Encoder 1 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")
        col2.plotly_chart(fig1)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc1'],
                                  name='enc1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc2'],
                                  name='enc2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc3'],
                                  name='enc3'))
        fig2.update_layout(title="Encoders 1, 2 e 3 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")        
        col3.plotly_chart(fig2)
        
    elif select == "enc2":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Encoder 2 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc1'],
                                  name='enc1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc2'],
                                  name='enc2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc3'],
                                  name='enc3'))
        fig2.update_layout(title="Encoders 1, 2 e 3 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")        
        col3.plotly_chart(fig2)
    elif select == "enc3":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Encoder 3 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc1'],
                                  name='enc1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc2'],
                                  name='enc2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['enc3'],
                                  name='enc3'))
        fig2.update_layout(title="Encoders 1, 2 e 3 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")        
        col3.plotly_chart(fig2)
    elif select == "strain1_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 1 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)        
    elif select == "strain2_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 2 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)
    elif select == "strain3_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 3 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)
    elif select == "strain4_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 4 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)
    elif select == "strain5_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 5 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)
    elif select == "strain6_filt":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Strain-Gauge 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain1_filt'],
                                  name='strain 1'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain2_filt'],
                                  name='strain 2'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain3_filt'],
                                  name='strain 3'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain4_filt'],
                                  name='strain 4'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain5_filt'],
                                  name='strain 5'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['strain6_filt'],
                                  name='strain 6'))
        fig2.update_layout(title="Strain-Gauges 1 até 6 ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Deformação")        
        col3.plotly_chart(fig2)
    elif select == "posx_trat":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Coordenada x ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['posx_trat'],
                                  name='posição x'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['posy_trat'],
                                  name='posição y'))
        fig2.update_layout(title="Coordenada x e y do efetuador",
                          xaxis_title='Tempo (s)',
                          yaxis_title='Posição')
        col3.plotly_chart(fig2)
    elif select == "posy_trat":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Coordenada y ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição")
        col2.plotly_chart(fig)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['posx_trat'],
                                  name='posição x'))
        fig2.add_trace(go.Scatter(x=x,
                                  y=df['posy_trat'],
                                  name='posição y'))
        fig2.update_layout(title="Coordenada x e y do efetuador",
                          xaxis_title='Tempo (s)',
                          yaxis_title='Posição')
        col3.plotly_chart(fig2)
    elif select == "alpha_trat":
        y = df[select]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode='lines',
                                 name=f'{select}'))
        fig.update_layout(title="Orientação alpha ao longo do tempo",
                          xaxis_title="Tempo (s)",
                          yaxis_title="Posição angular")
        col2.plotly_chart(fig)
    else:
        st.warning("Nenhum dado para visualizar ")

    with st.expander("Ver dados tabulares"):
        st.dataframe(df,use_container_width=True)    
with tab2:
    df_x = pd.read_csv("resultados_pred_x.csv")
    df_y = pd.read_csv("resultados_pred_y.csv")
    df_a = pd.read_csv("resultados_pred_alpha.csv")
    metricas_mse = pd.read_csv("resultados_mse.csv")
    metricas_mae = pd.read_csv("resultados_mae.csv")
    col1,col2,col3 = st.columns(3)
    col1.header("Visão geral: Simple RNN")
    col1.write("O Simple RNN (Recurrent Neural Network) é um tipo de arquitetura de \
               rede neural recorrente, projetada para trabalhar com dados sequenciais. \
               Ele é uma versão mais simples do que as redes recorrentes mais avançadas, \
               como LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Unit), e é \
               conhecido por ter dificuldades em lidar com problemas que envolvem \
               dependências temporais de longo prazo.")
    col1.write("- Simple RNN em Python:")
    col1.write("O Simple RNN possui células recorrentes que mantêm um estado interno, \
               permitindo que a rede mantenha uma 'memória' de informações anteriores \
               em uma sequência. Em Python, podemos implementar um Simple RNN usando \
               frameworks como TensorFlow ou Keras.")
    col1.write("Parâmetros importantes do Simples RNN:")
    col1.write("- units: Número de neurônios ou unidades na camada Simple RNN. \
               Controla a capacidade do modelo.")
    col1.write("- input_shape: Formato da entrada (número de etapas temporais, \
               número de características).")
    col1.write("- activation: Função de ativação aplicada às saídas da camada \
               (por exemplo, 'relu', 'tanh').")
    col2.image("imgs/simple_rnn_structure.jpg",use_column_width=True)
    code_simple_rnn = """model_simple_rnn = Sequential()
    model_simple_rnn.add(SimpleRNN(units=(64),
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])))
    model_simple_rnn.add(Dropout(0.2))
    model_simple_rnn.add(Dense(units=(64),activation='relu'))
    model_simple_rnn.add(Dropout(0.2))
    model_simple_rnn.add(Dense(1))    """
    col3.code(code_simple_rnn, language='python')
    col3.image("imgs/simple_rnn_model.jpg")

    col1, col2 = st.columns(2)
    col1.header("Resultados do treinamento")    
    col2.header("Gráficos comparativos")

    col1,col2,col3 = st.columns([1, 1.85, 1.75])
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.subheader("Métricas de desempenho - MSE")
    
    col1.write("Mean Squared Error - Predição posição x:")
    col1.write(metricas_mse['x'][0])
    col1.write("Mean Squared Error - Predição posição y:")
    col1.write(metricas_mse['y'][0])
    col1.write("Mean Squared Error - Predição orientação alpha:")
    col1.write(metricas_mse['alpha'][0])
    col1.subheader("Métricas de desempenho - MAE")
    col1.write("Mean Absolute Error - Predição posição x:")
    col1.write(metricas_mae['x'][0])
    col1.write("Mean Absolute Error - Predição posição y:")
    col1.write(metricas_mae['y'][0])
    col1.write("Mean Absolute Error - Predição orientação alpha:")
    col1.write(metricas_mae['alpha'][0])    
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['real x'],
                              name='posição x (real)'))
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['simple_rnn'],
                              name='posição x (simple rnn)'))
    fig1.update_layout(title="Comparação da trajetória 'x' real vs estimada (Modelo Simple RNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=1.0, traceorder='normal'))
    col2.plotly_chart(fig1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['real y'],
                              name='posição y (real)'))
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['simple_rnn'],
                              name='posição y (simple rnn)'))
    fig2.update_layout(title="Comparação da trajetória 'y' real vs estimada (Modelo Simple RNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col3.plotly_chart(fig2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['real alpha'],
                              name='posição y (real)'))
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['simple_rnn'],
                              name='posição y (simple rnn)'))
    fig3.update_layout(title="Comparação da posição angular 'alpha' real vs estimada (Modelo Simple RNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição angular',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col2.plotly_chart(fig3)
with tab3:
    df_x = pd.read_csv("resultados_pred_x.csv")
    df_y = pd.read_csv("resultados_pred_y.csv")
    df_a = pd.read_csv("resultados_pred_alpha.csv")
    metricas_mse = pd.read_csv("resultados_mse.csv")
    metricas_mae = pd.read_csv("resultados_mae.csv")
    col1,col2,col3 = st.columns(3)
    col1.header("Visão geral: LSTM")
    col1.write("Long Short-Term Memory (LSTM) é um tipo de arquitetura de rede neural \
               recorrente projetada para superar as limitações de dependências temporais\
                de longo prazo do Simple RNN (Recurrent Neural Network). O LSTM é capaz \
               de aprender dependências temporais mais complexas e é amplamente utilizado \
               em tarefas que envolvem sequências, como previsão de séries temporais, \
               processamento de linguagem natural e reconhecimento de fala.")
    col1.write("- LSTM em Python:")
    col1.write("Podemos implementar uma rede LSTM em Python usando frameworks como \
               TensorFlow ou Keras.")
    col1.write("Parâmetros importantes do LSTM:")
    col1.write("- units: Número de unidades ou neurônios na camada LSTM. Controla a \
               capacidade do modelo.")
    col1.write("- return_sequences: Se deve retornar a sequência completa ou apenas a \
               última saída. Usado quando você empilha camadas LSTM.")
    col1.write("- input_shape: Formato da entrada (número de etapas temporais, número \
               de características).")
    col1.write("- activation: Função de ativação aplicada às saídas da camada.")
    col2.image("imgs/lstm_structre.jpg",use_column_width=True)
    code_lstm = """model_lstm = Sequential()
    model_lstm.add(LSTM(units=(64),
            activation='relu',
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])))
    model_lstm.add(Dropout(0.2)) 
    model_lstm.add(LSTM(units=(64),
            activation='relu',
            return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=(64), activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(1))"""
    col3.code(code_lstm,language='python')
    col3.image("imgs/lstm_model.jpg")

    col1, col2 = st.columns(2)
    col1.header("Resultados do treinamento")    
    col2.header("Gráficos comparativos")

    col1,col2,col3 = st.columns([1, 1.85, 1.75])
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.subheader("Métricas de desempenho - MSE")
    
    col1.write("Mean Squared Error - Predição posição x:")
    col1.write(metricas_mse['x'][1])
    col1.write("Mean Squared Error - Predição posição y:")
    col1.write(metricas_mse['y'][1])
    col1.write("Mean Squared Error - Predição orientação alpha:")
    col1.write(metricas_mse['alpha'][1])
    col1.subheader("Métricas de desempenho - MAE")
    col1.write("Mean Absolute Error - Predição posição x:")
    col1.write(metricas_mae['x'][1])
    col1.write("Mean Absolute Error - Predição posição y:")
    col1.write(metricas_mae['y'][1])
    col1.write("Mean Absolute Error - Predição orientação alpha:")
    col1.write(metricas_mae['alpha'][1]) 

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['real x'],
                              name='posição x (real)'))
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['lstm'],
                              name='posição x (lstm)'))
    fig1.update_layout(title="Comparação da trajetória 'x' real vs estimada (Modelo LSTM)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=1.0, traceorder='normal'))
    col2.plotly_chart(fig1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['real y'],
                              name='posição y (real)'))
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['lstm'],
                              name='posição y (lstm)'))
    fig2.update_layout(title="Comparação da trajetória 'y' real vs estimada (Modelo LSTM)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col3.plotly_chart(fig2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['real alpha'],
                              name='posição y (real)'))
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['lstm'],
                              name='posição y (lstm)'))
    fig3.update_layout(title="Comparação da posição angular 'alpha' real vs estimada (Modelo LSTM)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição angular',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col2.plotly_chart(fig3)

with tab4:
    df_x = pd.read_csv("resultados_pred_x.csv")
    df_y = pd.read_csv("resultados_pred_y.csv")
    df_a = pd.read_csv("resultados_pred_alpha.csv")
    metricas_mse = pd.read_csv("resultados_mse.csv")
    metricas_mae = pd.read_csv("resultados_mae.csv")
    col1,col2,col3 = st.columns(3)
    col1.header("Visão geral: GRU")
    col1.write("Gate Recurrent Unit (GRU) é outro tipo de arquitetura de rede neural \
               recorrente, semelhante ao LSTM (Long Short-Term Memory), projetada para\
                resolver alguns dos desafios associados ao treinamento de redes \
               recorrentes. Assim como o LSTM, o GRU é especialmente eficaz em tarefas \
               que envolvem dependências temporais de longo prazo, como previsão de \
               séries temporais e processamento de linguagem natural.")
    col1.write("- GRU em Python:")
    col1.write("O GRU é implementado de maneira semelhante ao LSTM em Python usando \
               frameworks como TensorFlow ou Keras.")
    col1.write("Parâmetros importantes do GRU:")
    col1.write("- units: Número de unidades ou neurônios na camada GRU. Controla a \
               capacidade do modelo.")
    col1.write("- return_sequences: Se deve retornar a sequência completa ou apenas a \
               última saída. Usado quando você empilha camadas GRU.")
    col1.write("- input_shape: Formato da entrada (número de etapas temporais, \
               número de características).")
    col1.write("- activation: Função de ativação aplicada às saídas da camada.")
    col1.write("O GRU é uma variação mais simplificada do LSTM, mantendo eficácia na\
                aprendizagem de dependências temporais de longo prazo.")
    col2.image("imgs/gru_structure.jpg",use_column_width=True)
    col2.write("O GRU possui menos parâmetros do que o LSTM, o que pode ser benéfico \
               em termos de eficiência computacional e treinamento mais rápido.")
    col2.write("O GRU usa um mecanismo de portas mais simples em comparação com o LSTM, \
               o que pode ser vantajoso em certos cenários.")
    code_gru = """model_gru = Sequential()
    model_gru.add(GRU(units=(64),
            activation='relu',
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])))
    model_gru.add(Dropout(0.2))
    model_gru.add(GRU(units=(64),
            activation='relu',
            return_sequences=False))
    model_gru.add(Dropout(0.2))
    model_gru.add(Dense(64, activation = 'relu'))
    model_gru.add(Dropout(0.2))
    model_gru.add(Dense(1))"""
    col3.code(code_gru, language='python')
    col3.image("imgs/gru_model.jpg") 
    
    col1, col2 = st.columns(2)
    col1.header("Resultados do treinamento")    
    col2.header("Gráficos comparativos")

    col1,col2,col3 = st.columns([1, 1.85, 1.75])
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.subheader("Métricas de desempenho - MSE")
    
    col1.write("Mean Squared Error - Predição posição x:")
    col1.write(metricas_mse['x'][2])
    col1.write("Mean Squared Error - Predição posição y:")
    col1.write(metricas_mse['y'][2])
    col1.write("Mean Squared Error - Predição orientação alpha:")
    col1.write(metricas_mse['alpha'][2])
    col1.subheader("Métricas de desempenho - MAE")
    col1.write("Mean Absolute Error - Predição posição x:")
    col1.write(metricas_mae['x'][2])
    col1.write("Mean Absolute Error - Predição posição y:")
    col1.write(metricas_mae['y'][2])
    col1.write("Mean Absolute Error - Predição orientação alpha:")
    col1.write(metricas_mae['alpha'][2]) 

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['real x'],
                              name='posição x (real)'))
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['gru'],
                              name='posição x (gru)'))
    fig1.update_layout(title="Comparação da trajetória 'x' real vs estimada (Modelo GRU)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=1.0, traceorder='normal'))
    col2.plotly_chart(fig1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['real y'],
                              name='posição y (real)'))
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['gru'],
                              name='posição y (gru)'))
    fig2.update_layout(title="Comparação da trajetória 'y' real vs estimada (Modelo GRU)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col3.plotly_chart(fig2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['real alpha'],
                              name='posição y (real)'))
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['gru'],
                              name='posição y (gru)'))
    fig3.update_layout(title="Comparação da posição angular 'alpha' real vs estimada (Modelo GRU)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição angular',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col2.plotly_chart(fig3)

with tab5:
    df_x = pd.read_csv("resultados_pred_x.csv")
    df_y = pd.read_csv("resultados_pred_y.csv")
    df_a = pd.read_csv("resultados_pred_alpha.csv")
    metricas_mse = pd.read_csv("resultados_mse.csv")
    metricas_mae = pd.read_csv("resultados_mae.csv")
    col1,col2,col3 = st.columns(3)
    col1.header("Visão geral: CNN")
    col1.write("Convolutional Neural Network (CNN) é uma arquitetura de rede neural \
               projetada para processar dados com estrutura de grade, como imagens. \
               Ela é amplamente utilizada em tarefas relacionadas a visão computacional,\
                como reconhecimento de padrões, classificação de imagens e segmentação de \
               objetos.")
    col1.write("Componentes importantes da CNN:")
    col1.write("- Camada de Convolução: Detecta padrões locais na entrada.")
    col1.write("- - Parâmetros importantes: filters (número de filtros), kernel_size \
               (tamanho do filtro), activation (função de ativação)")
    col1.write("- Camada de Pooling: Reduz a dimensionalidade da representação e o custo \
               computacional.")
    col1.write("- - Parâmetros importantes: pool_size (tamanho da janela de pooling)")
    col1.write("- Camada Densa: Camadas totalmente conectadas para a saída final.")
    col1.write("- - Parâmetros importantes: units (número de neurônios), activation \
               (função de ativação)")
    col1.write("- Camada Flatten: Transforma a matriz de entrada em um vetor unidimensional \
               antes das camadas densas.")
    col2.image("imgs/cnn_structure1.png")
    col2.image("imgs/cnn_structure2.png")
    code_cnn = """model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64,
                kernel_size=5,
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(100, activation='relu'))
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Dense(1))"""      
    col3.code(code_cnn, language='python')   
    col3.image("imgs/cnn_model.jpg")

    col1, col2 = st.columns(2)
    col1.header("Resultados do treinamento")    
    col2.header("Gráficos comparativos")

    col1,col2,col3 = st.columns([1, 1.85, 1.75])
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.write(" ")
    col1.subheader("Métricas de desempenho - MSE")
    
    col1.write("Mean Squared Error - Predição posição x:")
    col1.write(metricas_mse['x'][3])
    col1.write("Mean Squared Error - Predição posição y:")
    col1.write(metricas_mse['y'][3])
    col1.write("Mean Squared Error - Predição orientação alpha:")
    col1.write(metricas_mse['alpha'][3])
    col1.subheader("Métricas de desempenho - MAE")
    col1.write("Mean Absolute Error - Predição posição x:")
    col1.write(metricas_mae['x'][3])
    col1.write("Mean Absolute Error - Predição posição y:")
    col1.write(metricas_mae['y'][3])
    col1.write("Mean Absolute Error - Predição orientação alpha:")
    col1.write(metricas_mae['alpha'][3]) 

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['real x'],
                              name='posição x (real)'))
    fig1.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['cnn'],
                              name='posição x (cnn)'))
    fig1.update_layout(title="Comparação da trajetória 'x' real vs estimada (Modelo CNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=1.0, traceorder='normal'))
    col2.plotly_chart(fig1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['real y'],
                              name='posição y (real)'))
    fig2.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['cnn'],
                              name='posição y (cnn)'))
    fig2.update_layout(title="Comparação da trajetória 'y' real vs estimada (Modelo CNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col3.plotly_chart(fig2)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['real alpha'],
                              name='posição y (real)'))
    fig3.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['cnn'],
                              name='posição y (cnn)'))
    fig3.update_layout(title="Comparação da posição angular 'alpha' real vs estimada (Modelo CNN)",
                       xaxis_title='Tempo (s)',
                       yaxis_title='Posição angular',
                       legend=dict(x=0.0, y=0.0, traceorder='normal'))
    col2.plotly_chart(fig3)

with tab6:
    df_x = pd.read_csv("resultados_pred_x.csv")
    df_y = pd.read_csv("resultados_pred_y.csv")
    df_a = pd.read_csv("resultados_pred_alpha.csv")
    metricas_mse = pd.read_csv("resultados_mse.csv")
    metricas_mae = pd.read_csv("resultados_mae.csv")
    st.header("Métricas dos 4 modelos para cada output separadamente")
    categorias = ["Simple RNN","LSTM", "GRU", "CNN"]
    mse_valores_x = [metricas_mse['x'][0],
               metricas_mse['x'][1],
               metricas_mse['x'][2],
               metricas_mse['x'][3]]
    mse_valores_y = [metricas_mse['y'][0],
               metricas_mse['y'][1],
               metricas_mse['y'][2],
               metricas_mse['y'][3]]
    mse_valores_a = [metricas_mse['alpha'][0],
               metricas_mse['alpha'][1],
               metricas_mse['alpha'][2],
               metricas_mse['alpha'][3]]
    col1, col2, col3 = st.columns([1,1,1.5])
    fig1 = go.Figure(data=go.Bar(x=categorias,y=mse_valores_x, width=0.5))
    fig1.update_layout(title="MSE da estimativa dos modelos para 'x'")
    col1.plotly_chart(fig1,use_container_width=True)

    fig2 = go.Figure(data=go.Bar(x=categorias,y=mse_valores_y, width=0.5))
    fig2.update_layout(title="MSE da estimativa dos modelos para 'y'")
    col1.plotly_chart(fig2,use_container_width=True)

    fig3 = go.Figure(data=go.Bar(x=categorias,y=mse_valores_a, width=0.5))
    fig3.update_layout(title="MSE da estimativa dos modelos para 'alpha'")
    col1.plotly_chart(fig3, use_container_width=True)

    mae_valores_x = [metricas_mae['x'][0],
               metricas_mae['x'][1],
               metricas_mae['x'][2],
               metricas_mae['x'][3]]
    mae_valores_y = [metricas_mae['y'][0],
               metricas_mae['y'][1],
               metricas_mae['y'][2],
               metricas_mae['y'][3]]
    mae_valores_a = [metricas_mae['alpha'][0],
               metricas_mae['alpha'][1],
               metricas_mae['alpha'][2],
               metricas_mae['alpha'][3]]
    
    fig4 = go.Figure(data=go.Bar(x=categorias,y=mae_valores_x, width=0.5))
    fig4.update_layout(title="MAE da estimativa dos modelos para 'x'")
    col2.plotly_chart(fig4,use_container_width=True)

    fig5 = go.Figure(data=go.Bar(x=categorias,y=mae_valores_y, width=0.5))
    fig5.update_layout(title="MAE da estimativa dos modelos para 'y'")
    col2.plotly_chart(fig5, use_container_width=True)

    fig6 = go.Figure(data=go.Bar(x=categorias,y=mae_valores_a, width=0.5))
    fig6.update_layout(title="MAE da estimativa dos modelos para 'alpha'")
    col2.plotly_chart(fig6, use_container_width=True)

    figx = go.Figure()
    figx.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['real x'],
                              name='real x'))
    figx.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['simple_rnn'],
                              name='Simple RNN'))
    figx.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['lstm'],
                              name='LSTM'))
    figx.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['gru'],
                              name='GRU'))
    figx.add_trace(go.Scatter(x=df_x['tempo'],
                              y=df_x['cnn'],
                              name='CNN'))
    figx.update_layout(title="Comparação da posição x estimada (todos modelos) vs real",
                       xaxis_title="Tempo (s)",
                       yaxis_title="Posição")
    col3.plotly_chart(figx, use_container_width=True)

    figy = go.Figure()
    figy.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['real y'],
                              name='real y'))
    figy.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['simple_rnn'],
                              name='Simple RNN'))
    figy.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['lstm'],
                              name='LSTM'))
    figy.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['gru'],
                              name='GRU'))
    figy.add_trace(go.Scatter(x=df_y['tempo'],
                              y=df_y['cnn'],
                              name='CNN'))
    figy.update_layout(title="Comparação da posição y estimada (todos modelos) vs real",
                       xaxis_title="Tempo (s)",
                       yaxis_title="Posição")
    col3.plotly_chart(figy, use_container_width=True)
    
    figa = go.Figure()
    figa.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['real alpha'],
                              name='real alpha'))
    figa.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['simple_rnn'],
                              name='Simple RNN'))
    figa.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['lstm'],
                              name='LSTM'))
    figa.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['gru'],
                              name='GRU'))
    figa.add_trace(go.Scatter(x=df_a['tempo'],
                              y=df_a['cnn'],
                              name='CNN'))
    figa.update_layout(title="Comparação da posição alpha estimada (todos modelos) vs real",
                       xaxis_title="Tempo (s)",
                       yaxis_title="Posição angular")
    col3.plotly_chart(figa, use_container_width=True)
