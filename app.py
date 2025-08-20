import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import altair as alt

def main():
    st.set_page_config(page_title="Dashboard Ventas - Supertienda", layout="wide")

    st.title("📊 Dashboard Interactivo de Ventas - Supertienda")
    st.markdown("""
    Explora los datos históricos, visualiza KPIs y realiza predicciones de ventas.
    """)

    # ===============================
    # Carga de dataset
    # ===============================
    file = st.file_uploader("📂 Sube tu dataset (Excel o CSV, versión en español únicamente)", type=["xls","xlsx","csv"])

    if file is not None:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8")
        else:
            df = pd.read_excel(file, engine="openpyxl")

        # ===============================
        # Preprocesamiento
        # ===============================
        df['Fecha del pedido'] = pd.to_datetime(df['Fecha del pedido'])
        df['Fecha de envío'] = pd.to_datetime(df['Fecha de envío'])
        df['Año'] = df['Fecha del pedido'].dt.year
        df['Mes'] = df['Fecha del pedido'].dt.month

        categorical_cols = ['Categoría', 'Subcategoría', 'Región', 'Segmento']
        encoders = {}
        encoded_options = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            encoded_options[col] = {name: code for name, code in zip(le.classes_, le.transform(le.classes_))}

        # ===============================
        # KPIs
        # ===============================
        st.subheader("📌 KPIs de ventas")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas totales", f"${df['Ventas'].sum():,.2f}")
        col2.metric("Ventas promedio", f"${df['Ventas'].mean():,.2f}")
        col3.metric("Cantidad total", int(df['Cantidad'].sum()))

        # ===============================
        # Gráficos históricos
        # ===============================
        st.subheader("📈 Tendencias de ventas")

        ventas_mes = df.groupby(['Año','Mes'])['Ventas'].sum().reset_index()
        chart1 = alt.Chart(ventas_mes).mark_line(point=True).encode(
            x=alt.X('Mes:O', title='Mes'),
            y=alt.Y('Ventas', title='Ventas Totales'),
            color='Año:N',
            tooltip=['Año','Mes','Ventas']
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)

        ventas_cat = df.groupby('Categoría')['Ventas'].sum().reset_index()
        chart2 = alt.Chart(ventas_cat).mark_bar().encode(
            x='Categoría:N',
            y='Ventas',
            tooltip=['Categoría','Ventas']
        )
        st.altair_chart(chart2, use_container_width=True)

        # ===============================
        # Entrenamiento del modelo
        # ===============================
        X = df[['Año', 'Mes', 'Categoría', 'Subcategoría', 'Región', 'Segmento', 'Cantidad', 'Descuento']]
        y = df['Ventas']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        st.success("✅ Modelo Random Forest entrenado")

        # ===============================
        # Simulador de predicción
        # ===============================
        st.subheader("🛠️ Simulador de predicciones")
        col1, col2 = st.columns(2)

        with col1:
            año = st.number_input("Año", min_value=int(df['Año'].min()), max_value=int(df['Año'].max()), value=int(df['Año'].max()))
            mes = st.slider("Mes", 1, 12, 1)
            categoria = st.selectbox("Categoría", list(encoded_options['Categoría'].keys()))
            subcategoria = st.selectbox("Subcategoría", list(encoded_options['Subcategoría'].keys()))

        with col2:
            region = st.selectbox("Región", list(encoded_options['Región'].keys()))
            segmento = st.selectbox("Segmento", list(encoded_options['Segmento'].keys()))
            cantidad = st.number_input("Cantidad", min_value=1, value=1)
            descuento = st.number_input("Descuento", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        input_data = pd.DataFrame({
            'Año': [año],
            'Mes': [mes],
            'Categoría': [encoded_options['Categoría'][categoria]],
            'Subcategoría': [encoded_options['Subcategoría'][subcategoria]],
            'Región': [encoded_options['Región'][region]],
            'Segmento': [encoded_options['Segmento'][segmento]],
            'Cantidad': [cantidad],
            'Descuento': [descuento]
        })

        pred = model.predict(input_data)[0]
        st.metric(label="💰 Predicción de ventas", value=f"${pred:,.2f}")

        # ===============================
        # Importancia de variables
        # ===============================
        st.subheader("🔍 Importancia de variables")
        importancia = pd.DataFrame({
            "Variable": X.columns,
            "Importancia": model.feature_importances_
        }).sort_values(by="Importancia", ascending=False)
        st.bar_chart(importancia.set_index("Variable"))

if __name__ == "__main__":
    main()
