import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import joblib


DATA_URL = "automobile_data.csv"
MODEL_URL = "logmodel.sav"

def load_data():
    data = pd.read_csv(DATA_URL)

    # Higienizar dados faltantes
    data = data.replace('?', np.nan)

    # Converter columnas numericas que eram string devido ao '?'
    data['normalized-losses'] = pd.to_numeric(data['normalized-losses'])
    data['bore'] = pd.to_numeric(data['bore'])
    data['stroke'] = pd.to_numeric(data['stroke'])
    data['horsepower'] = pd.to_numeric(data['horsepower'])
    data['peak-rpm'] = pd.to_numeric(data['peak-rpm'])
    data['price'] = pd.to_numeric(data['price'])

    # Preencher valores faltantes
    data['normalized-losses'].fillna(data['normalized-losses'].mean(), inplace = True)
    data['num-of-doors'].fillna(data['num-of-doors'].mode()[0], inplace = True)
    data['bore'].fillna(data['bore'].mean(), inplace = True)
    data['stroke'].fillna(data['stroke'].mean(), inplace = True)
    data['horsepower'].fillna(data['horsepower'].mean(), inplace = True)
    data['peak-rpm'].fillna(data['peak-rpm'].mean(), inplace = True)
    data['price'].fillna(data['price'].mean(), inplace = True)

    # Seleção de colunas relevantes
    x = data[['horsepower', 'curb-weight', 'city-mpg', 'highway-mpg', 'wheel-base', 'engine-size', 'compression-ratio', 'peak-rpm', 'length', 'width', 'height']]

    y = data.iloc[:25]
    return (x,y)

def user_input_features(xCol):
    data = {}
    for col in xCol.columns.to_list():
        minimo = np.float32(xCol[col].min()).item()
        maximo = np.float32(xCol[col].max()).item()
        media = np.float32(xCol[col].mean()).item()

        data[col] = st.sidebar.slider(col, minimo, maximo, media)
    return pd.DataFrame(data, index=[0])

def main():
    X_carros, y_carros = load_data()

    st.write("""
    # Predição de preços de carro
    # """)
    st.write('---')


    model = joblib.load(MODEL_URL)
    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Escolha de paramentros para Predição')

    df = user_input_features(X_carros)

    st.sidebar.markdown("""
    Foi optado por manter diversos campos para que o modelo treinado pudesse ter
    uma acurácia satisfatória. Os campos com maior peso na predição são:
    - **horsepower**
    - **curb-weight**
    - **city-mpg**
    - **highway-mpg**
    """)

    st.header('Parametros especificados')
    st.write(df)
    st.write('---')

    prediction = model.predict(df)
    price = ['Baixo', 'Médio', 'Alto'][prediction[0]]

    st.header('Preço Previsto')
    st.write(price)
    st.write('---')

if __name__=='__main__':
    main()
