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
            
            # Limpieza inicial obligatoria según las instrucciones del caso
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(columns=['Unnamed: 0'])
                
            if 'Period Ending' in self.df.columns:
                self.df['Period Ending'] = pd.to_datetime(self.df['Period Ending'], errors='coerce')
                
            return True
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            return False

def module_home():
    st.title("📊 Análisis Exploratorio de Datos: New York Stock Exchange")
    st.markdown("""
    Este proyecto aplicativo realiza un Análisis Exploratorio de Datos (EDA) sobre los estados financieros históricos de empresas listadas en la Bolsa de Nueva York. El objetivo principal es evaluar ingresos, rentabilidad, liquidez, activos y flujos de efectivo, garantizando comparaciones precisas mediante una correcta escala de valores monetarios.
    
    **Datos del Autor**
    * **Nombre:** Zander Lutty Delgado Urbina
    * **Curso:** Especialización en Python for Analytics
    * **Año:** 2026
    
    **Tecnologías Utilizadas**
    * Python, Pandas, NumPy
    * Matplotlib, Seaborn
    * Streamlit (Interfaz interactiva)
    """)

def module_data_load():
    st.header("📂 Carga de Dataset")
    st.write("Sube el archivo `New York Stock Exchange.csv` para comenzar.")
    
    uploaded_file = st.file_uploader("Selecciona el archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        processor = DataProcessor()
        if processor.load_data(uploaded_file):
            # Guardamos la instancia de POO en session_state para persistencia entre módulos
            st.session_state['data_processor'] = processor
            
            st.success("✅ Archivo cargado y validado correctamente.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filas", processor.df.shape[0])
            with col2:
                st.metric("Columnas", processor.df.shape[1])
                
            st.subheader("Vista previa del Dataset")
            st.dataframe(processor.df.head(), use_container_width=True)
    else:
        st.info("Esperando la carga del archivo...")
        if 'data_processor' in st.session_state:
            del st.session_state['data_processor']

def main():
    st.sidebar.title("Menú Principal")
    menu = ["1. Home", "2. Carga de Dataset", "3. Análisis EDA", "4. Conclusiones"]
    choice = st.sidebar.radio("Navegación", menu)
    
    if choice == "1. Home":
        module_home()
    elif choice == "2. Carga de Dataset":
        module_data_load()
    elif choice == "3. Análisis EDA":
        st.header("📈 Análisis Exploratorio de Datos")
        if 'data_processor' not in st.session_state:
            st.warning("⚠️ Bloqueo de seguridad: Debes cargar el archivo en el Módulo 2 antes de ejecutar cualquier análisis.")
        else:
            st.info("Dataset validado. Aquí programaremos los 10 ítems en la siguiente fase.")
    elif choice == "4. Conclusiones":
        st.header("💡 Conclusiones Finales")
        st.info("Esta sección se nutrirá de los hallazgos del EDA.")

if __name__ == "__main__":
    main()
