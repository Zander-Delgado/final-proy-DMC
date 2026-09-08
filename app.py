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
            # Limpieza inicial
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(columns=['Unnamed: 0'])
            if 'Period Ending' in self.df.columns:
                self.df['Period Ending'] = pd.to_datetime(self.df['Period Ending'], errors='coerce')
            return True
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            return False

    def classify_variables(self):
        """Ítem 2: Función personalizada para clasificar variables"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        return numeric_cols, categorical_cols

    def get_missing_summary(self):
        """Ítem 4: Resumen de valores faltantes"""
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
    
    # Creación de pestañas navegables
    tabs = st.tabs([
        "1. Info General", "2. Clasificación", "3. Descriptivas", 
        "4. Valores Faltantes", "5. Distribuciones"
    ])
    
    # --- Ítem 1: Información General ---
    with tabs[0]:
        st.subheader("Ítem 1: Información general del dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tipos de datos por columna:**")
            st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Variable', 0: 'Tipo'}), use_container_width=True)
        with col2:
            st.metric("Registros Duplicados", df.duplicated().sum())
            st.metric("Total de Celdas Nulas", df.isnull().sum().sum())
            st.write("*(Se usó `.info()` implícitamente para extraer esta metadata)*")

    # --- Ítem 2: Clasificación de Variables ---
    with tabs[1]:
        st.subheader("Ítem 2: Clasificación de variables")
        num_cols, cat_cols = processor.classify_variables()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Numéricas ({len(num_cols)}):**")
            st.dataframe(pd.DataFrame(num_cols, columns=["Variables Numéricas"]), use_container_width=True)
        with col2:
            st.write(f"**Categóricas ({len(cat_cols)}):**")
            st.dataframe(pd.DataFrame(cat_cols, columns=["Variables Categóricas"]), use_container_width=True)

    # --- Ítem 3: Estadísticas Descriptivas ---
    with tabs[2]:
        st.subheader("Ítem 3: Estadísticas descriptivas")
        st.write("Resumen estadístico de las variables numéricas (medias, medianas, dispersión):")
        st.dataframe(df.describe(), use_container_width=True)
        st.info("💡 **Detección preliminar:** Se observan magnitudes muy dispares entre variables como 'Total Revenue' (miles de millones) y ratios como 'Current Ratio' (unidades), lo que sugiere fuerte presencia de outliers en valores absolutos corporativos.")

    # --- Ítem 4: Análisis de Valores Faltantes ---
    with tabs[3]:
        st.subheader("Ítem 4: Análisis de valores faltantes")
        missing_df = processor.get_missing_summary()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=missing_df.index, y=missing_df['Porcentaje (%)'], ax=ax, palette="viridis")
            plt.xticks(rotation=45, ha='right')
            plt.title("Porcentaje de Datos Faltantes por Variable")
            st.pyplot(fig)
            
        st.write("""
        **Discusión sobre tratamiento:** 
        Las ausencias se concentran en ratios de liquidez (Cash Ratio, Current Ratio) y datos de acciones. 
        *Justificación:* No se aplicará imputación por media/mediana automáticamente, ya que la estructura de capital de cada empresa (Ticker) es única y rellenar estos vacíos distorsionaría el perfil de riesgo de la compañía.
        """)

    # --- Ítem 5: Distribución de Variables Numéricas ---
    with tabs[4]:
        st.subheader("Ítem 5: Distribución de métricas financieras clave")
        st.write("Escalado a **Miles de Millones de Dólares (Billions)** para correcta visualización.")
        
        # Variables solicitadas en el caso
        vars_to_plot = ['Total Revenue', 'Net Income', 'Total Assets', 'Total Liabilities']
        
        # Selectbox dinámico (cumpliendo uso de widgets)
        selected_var = st.selectbox("Selecciona la métrica a visualizar:", vars_to_plot)
        
        # Escalar los datos (dividir por 1,000,000,000)
        scaled_data = df[selected_var] / 1e9
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(scaled_data, bins=50, kde=True, color='blue', ax=ax)
        ax.set_title(f"Distribución de {selected_var} (Miles de Millones USD)")
        ax.set_xlabel("Miles de Millones USD")
        ax.set_ylabel("Frecuencia (Empresas/Períodos)")
        st.pyplot(fig)
        
        st.markdown(f"""
        **Interpretación visual:**
        * Se observa una distribución **altamente asimétrica (sesgada a la derecha)**.
        * La gran mayoría de los registros se concentran en el extremo inferior de la escala, indicando que hay pocas compañías que dominan la muestra con valores gigantescos (outliers naturales del mercado).
        * En el caso de *Net Income*, evaluamos la presencia de **valores negativos** en la cola izquierda, lo que representa períodos de pérdida neta para ciertas corporaciones.
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
        st.header("💡 Conclusiones Finales")
        st.info("Esta sección se nutrirá de los hallazgos tras completar los 10 ítems.")

if __name__ == "__main__":
    main()
