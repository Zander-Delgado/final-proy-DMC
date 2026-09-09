import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NYSE Data Analysis", layout="wide", page_icon="📈")

class DataProcessor:
    """Clase principal orientada a objetos (POO) para encapsular la lógica de datos."""
    def __init__(self):
        self.df = None

    def load_data(self, file):
        try:
            self.df = pd.read_csv(file)
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(columns=['Unnamed: 0'])
            if 'Period Ending' in self.df.columns:
                self.df['Period Ending'] = pd.to_datetime(self.df['Period Ending'], errors='coerce')
            return True
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            return False

    def classify_variables(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_cols, categorical_cols

    def get_missing_summary(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Valores Nulos': missing, 'Porcentaje (%)': missing_pct})
        return missing_df[missing_df['Valores Nulos'] > 0].sort_values(by='Valores Nulos', ascending=False)

def module_home():
    st.title("📊 Análisis Exploratorio de Datos: New York Stock Exchange")
    st.markdown("""
    Este proyecto aplicativo realiza un Análisis Exploratorio de Datos (EDA) sobre los estados financieros históricos de empresas listadas en la Bolsa de Nueva York. El objetivo principal es evaluar ingresos, rentabilidad, liquidez, activos y flujos de efectivo, garantizando comparaciones precisas mediante una correcta escala de valores monetarios.
    
    **Datos del Autor**
    * **Nombre:** Zander Lutty Delgado Urbina
    * **Curso:** Especialización en Python for Analytics
    * **Año:** 2026
    
    **Tecnologías Utilizadas**
    * Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit
    """)

def module_data_load():
    st.header("📂 Carga de Dataset")
    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        processor = DataProcessor()
        if processor.load_data(uploaded_file):
            st.session_state['data_processor'] = processor
            st.success("✅ Archivo cargado y validado correctamente.")
            col1, col2 = st.columns(2)
            col1.metric("Total Filas", processor.df.shape[0])
            col2.metric("Total Columnas", processor.df.shape[1])
            st.dataframe(processor.df.head(), use_container_width=True)
    else:
        st.info("Esperando la carga del archivo...")
        if 'data_processor' in st.session_state:
            del st.session_state['data_processor']

def module_eda():
    st.header("📈 Análisis Exploratorio de Datos (EDA)")
    
    if 'data_processor' not in st.session_state:
        st.warning("⚠️ Bloqueo de seguridad: Debes cargar el archivo en el Módulo 2 primero.")
        return
        
    processor = st.session_state['data_processor']
    df = processor.df
    
    tabs = st.tabs([
        "1 a 3: General", "4: Nulos", "5: Distribuciones", 
        "6: Categóricas", "7: Num vs Cat", "8: Cat vs Cat", "9: Dinámico", "10: Hallazgos"
    ])
    
    # --- Ítems 1, 2 y 3: Resumen General ---
    with tabs[0]:
        st.subheader("Ítems 1, 2 y 3: Información General y Descriptiva")
        num_cols, cat_cols = processor.classify_variables()
        col1, col2, col3 = st.columns(3)
        col1.metric("Registros Duplicados", df.duplicated().sum())
        col2.metric("Variables Numéricas", len(num_cols))
        col3.metric("Variables Categóricas", len(cat_cols))
        st.write("**Estadísticas Descriptivas (Muestra):**")
        st.dataframe(df[['Total Revenue', 'Net Income', 'Total Assets', 'Total Liabilities']].describe(), use_container_width=True)

    # --- Ítem 4: Valores Faltantes ---
    with tabs[1]:
        st.subheader("Ítem 4: Análisis de valores faltantes")
        missing_df = processor.get_missing_summary()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=missing_df.index, y=missing_df['Porcentaje (%)'], ax=ax, palette="viridis")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    # --- Ítem 5: Distribuciones ---
    with tabs[2]:
        st.subheader("Ítem 5: Distribución de métricas financieras")
        var_to_plot = st.selectbox("Selecciona la métrica (Escala: Miles de Millones USD):", 
                                   ['Total Revenue', 'Net Income', 'Total Assets', 'Total Liabilities'])
        scaled_data = df[var_to_plot] / 1e9
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(scaled_data, bins=50, kde=True, color='royalblue', ax=ax)
        st.pyplot(fig)

    # --- Ítem 6: Análisis de Variables Categóricas ---
    with tabs[3]:
        st.subheader("Ítem 6: Análisis de variables categóricas")
        st.write("Conteo de reportes financieros emitidos por año fiscal (`For Year`).")
        year_counts = df['For Year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, palette="magma", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Ítem 7: Análisis Bivariado (Numérico vs Categórico) ---
    with tabs[4]:
        st.subheader("Ítem 7: Análisis Bivariado (Numérico vs Categórico)")
        st.write("Comparación de **Margen de Utilidad (Profit Margin)** para las 10 empresas con más registros.")
        top_tickers = df['Ticker Symbol'].value_counts().head(10).index
        filtered_df = df[df['Ticker Symbol'].isin(top_tickers)]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered_df, x='Ticker Symbol', y='Profit Margin', palette="Set2", ax=ax)
        st.pyplot(fig)

    # --- Ítem 8: Análisis Bivariado (Categórico vs Categórico) ---
    with tabs[5]:
        st.subheader("Ítem 8: Análisis Bivariado (Categórico vs Categórico)")
        st.write("Clasificación de Riesgo de Liquidez por Año Fiscal.")
        df_liq = df.dropna(subset=['Current Ratio', 'For Year']).copy()
        # Creación de categoría de liquidez
        df_liq['Riesgo Liquidez'] = np.where(df_liq['Current Ratio'] < 1.0, 'Alto (Ratio < 1)', 'Saludable (Ratio >= 1)')
        ct = pd.crosstab(df_liq['For Year'], df_liq['Riesgo Liquidez'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        ct.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=ax)
        plt.ylabel("Proporción (%)")
        st.pyplot(fig)

    # --- Ítem 9: Análisis Dinámico con Parámetros ---
    with tabs[6]:
        st.subheader("Ítem 9: Filtros y Análisis Dinámico")
        tickers_disp = df['Ticker Symbol'].unique()
        selected_tickers = st.multiselect("Selecciona Empresas (Ticker Symbol):", tickers_disp, default=['AAPL', 'MSFT', 'GOOGL'])
        
        if selected_tickers:
            min_rev = float(df['Total Revenue'].min())
            max_rev = float(df['Total Revenue'].max())
            rev_range = st.slider("Filtro por Rango de Ingresos (Total Revenue):", min_rev, max_rev, (min_rev, max_rev))
            
            dyn_df = df[(df['Ticker Symbol'].isin(selected_tickers)) & 
                        (df['Total Revenue'] >= rev_range[0]) & 
                        (df['Total Revenue'] <= rev_range[1])]
            
            st.write(f"Registros encontrados: {len(dyn_df)}")
            st.dataframe(dyn_df[['Ticker Symbol', 'Period Ending', 'Total Revenue', 'Net Income', 'Current Ratio']], use_container_width=True)

    # --- Ítem 10: Hallazgos Clave ---
    with tabs[7]:
        st.subheader("Ítem 10: Hallazgos Clave del EDA")
        st.markdown("""
        * **Concentración de la Muestra:** Existe un alto grado de asimetría positiva en los ingresos corporativos; unas pocas empresas dominan el volumen total de activos e ingresos.
        * **Faltantes Sistemáticos:** Los nulos no son aleatorios. Se concentran en ratios específicos (Current/Quick Ratio), lo que sugiere que ciertas industrias (ej. sector bancario) estructuran sus balances de manera distinta y no reportan liquidez tradicional.
        * **Márgenes de Utilidad:** Se identifican períodos de utilidad neta negativa (pérdidas), lo que hace indispensable evaluar el flujo de caja operativo como métrica complementaria para medir la salud real del negocio.
        """)

def module_conclusions():
    st.header("💡 Conclusiones Finales")
    st.markdown("""
    1. **Viabilidad y Riesgo de Liquidez:** La evaluación cruzada del *Current Ratio* revela que una proporción constante de empresas opera con índices inferiores a 1.0. Esto subraya la necesidad de auditorías profundas antes de establecer alianzas estratégicas o de proveeduría a largo plazo.
    2. **Estructura de Capital:** La disparidad entre *Total Assets* y *Total Liabilities* expone modelos de negocio con distintos niveles de apalancamiento, lo que afecta directamente el *After Tax ROE* (Rentabilidad sobre el patrimonio).
    3. **Impacto de Valores Extremos:** Las variables monetarias exhiben distribuciones fuertemente sesgadas hacia la derecha. Las decisiones gerenciales basadas en medias aritméticas pueden ser engañosas; es mandatorio el uso de medianas y percentiles para análisis comparativos.
    4. **Estacionalidad y Continuidad:** Los reportes anuales muestran una cobertura desigual dependiendo del *For Year*. Esto indica que los análisis de tendencias plurianuales deben estandarizarse para evitar sesgos por falta de datos históricos.
    5. **Efectividad del EDA en Python:** El uso de Pandas y Streamlit permitió consolidar 1,781 estados financieros en un tablero interactivo, transformando datos crudos en inteligencia de negocio de manera instantánea, un proceso altamente escalable para el monitoreo corporativo.
    """)

def main():
    st.sidebar.title("Menú Principal")
    menu = ["1. Home", "2. Carga de Dataset", "3. Análisis EDA", "4. Conclusiones"]
    choice = st.sidebar.radio("Navegación", menu)
    
    if choice == "1. Home":
        module_home()
    elif choice == "2. Carga de Dataset":
        module_data_load()
    elif choice == "3. Análisis EDA":
        module_eda()
    elif choice == "4. Conclusiones":
        module_conclusions()

if __name__ == "__main__":
    main()
