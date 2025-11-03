# 🌍 Sistema de Optimización de Reubicación de Colaboradores

## 📋 Descripción del Proyecto

Sistema integral de análisis geoespacial y optimización para la reubicación estratégica de colaboradores en una red de tiendas, minimizando distancias de desplazamiento e implementando el algoritmo **Top Trading Cycles (TTC)** para intercambios mutuamente beneficiosos.

Este proyecto de tesis aborda el problema del impacto negativo de las largas distancias de desplazamiento en la productividad laboral, proponiendo una solución tecnológica que optimiza la asignación de colaboradores a tiendas basándose en criterios de proximidad geográfica y preferencias mutuas.

## 🎯 Objetivos

- **Reducir tiempos de desplazamiento** de colaboradores mediante reubicación estratégica
- **Optimizar la productividad** minimizando el impacto negativo del commuting
- **Implementar intercambios justos** mediante el algoritmo TTC
- **Visualizar datos geoespaciales** para toma de decisiones informadas
- **Automatizar el proceso** de geocodificación y cálculo de distancias

## 🚀 Características Principales

### 📊 Análisis y Procesamiento
- **Geocodificación automática** de direcciones usando Google Maps API
- **Cálculo de distancias y tiempos** de desplazamiento en tiempo real
- **Limpieza inteligente** de direcciones colombianas
- **Procesamiento masivo** de datos de colaboradores

### 🔄 Algoritmo TTC (Top Trading Cycles)
- Implementación adaptada para reubicación laboral
- Identificación automática de ciclos de intercambio
- Cálculo de mejoras potenciales en distancia/tiempo
- Garantía de asignaciones Pareto-óptimas

### 📱 Aplicación Web Interactiva
- Dashboard en Streamlit con visualización en tiempo real
- Mapas interactivos con PyDeck
- Filtros dinámicos por ubicación, cargo y métricas
- Visualización de conexiones colaborador-tienda

### 📈 Métricas y KPIs
- Porcentaje de colaboradores con potencial de reubicación
- Reducción promedio en distancia/tiempo de desplazamiento
- Análisis de intercambios sugeridos
- Estadísticas por tienda y región

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python 3.9+** - Lenguaje principal
- **Pandas** - Manipulación y análisis de datos
- **NumPy** - Operaciones numéricas
- **GeoPy** - Geocodificación y cálculos geográficos
- **Google Maps API** - Servicios de geocodificación y distancias

### Frontend
- **Streamlit** - Framework de aplicación web
- **PyDeck** - Visualización de mapas 3D
- **Plotly** - Gráficos interactivos
- **Folium** - Mapas web interactivos

### Machine Learning & Análisis
- **Scikit-learn** - Análisis predictivo
- **CatBoost/XGBoost** - Modelos de gradient boosting
- **NLTK** - Procesamiento de lenguaje natural para direcciones

## 📁 Estructura del Proyecto

```
codigo_tesis_git/
│
├── app/                          # Aplicación web Streamlit
│   ├── main.py                  # Punto de entrada principal
│   ├── config.py                # Configuración y banner
│   ├── filters.py               # Filtros del sidebar
│   ├── layout.py                # Componentes de UI
│   ├── map_layers.py            # Capas del mapa
│   └── utils/                   # Utilidades de la app
│       ├── columnas.py          # Manejo de columnas
│       ├── coordenadas.py       # Funciones geográficas
│       └── limpieza_direccion.py # Limpieza de direcciones
│
├── scripts/                      # Scripts de procesamiento
│   ├── TTC.py                   # Implementación del algoritmo TTC
│   ├── ejecucion_final.ipynb   # Notebook principal de análisis
│   ├── limpieza_direcciones.py # Script de limpieza
│   ├── obtencion_coordenadas.py # Geocodificación
│   ├── obtencion_distancias_ttc.py # Cálculo de distancias
│   │
│   ├── data/                    # Directorios de datos
│   │   ├── raw/                 # Datos originales
│   │   ├── processed/           # Datos procesados
│   │   ├── coordenadas/         # Coordenadas geocodificadas
│   │   ├── distancias/          # Matrices de distancias
│   │   └── app/                 # Datos para la aplicación
│   │
│   └── utils/                   # Utilidades del backend
│       ├── data_utils.py        # Funciones de datos
│       ├── distancias.py        # Cálculo de distancias
│       ├── filtros_exclusion.py # Filtros de exclusión
│       └── obtencion_coordenadas.py # Utilidades de geocodificación
│
└── requirements.txt             # Dependencias del proyecto
```

## 🔧 Instalación

### Prerrequisitos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Cuenta de Google Cloud con Maps API habilitada

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/codigo_tesis_git.git
cd codigo_tesis_git
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
# Crear archivo .env en la raíz del proyecto
echo "GOOGLE_MAPS_API_KEY=tu_api_key_aqui" > .env
```

## 📊 Uso del Sistema

### 1. Preparación de Datos

```python
# Ejecutar el notebook de procesamiento
jupyter notebook scripts/ejecucion_final.ipynb
```

### 2. Limpieza de Direcciones

```bash
python scripts/limpieza_direcciones.py --input data/raw/maestro.xlsx --output data/processed/
```

### 3. Geocodificación

```bash
python scripts/obtencion_coordenadas.py --batch-size 100
```

### 4. Cálculo de Distancias y TTC

```bash
python scripts/obtencion_distancias_ttc.py --mode driving
```

### 5. Lanzar Aplicación Web

```bash
streamlit run app/main.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📈 Algoritmo TTC - Detalles Técnicos

El algoritmo Top Trading Cycles implementado sigue estos pasos:

1. **Inicialización**: Cada colaborador apunta a su tienda preferida más cercana
2. **Identificación de ciclos**: Búsqueda de ciclos en el grafo de preferencias
3. **Asignación**: Los colaboradores en ciclos intercambian posiciones
4. **Iteración**: El proceso se repite hasta que no quedan más ciclos

```python
from TTC import TTCReubicacion

# Ejemplo de uso
ttc = TTCReubicacion(dataframe_colaboradores)
resultados, resumen_ciclos = ttc.ejecutar(verbose=True)
```

## 📊 Métricas de Impacto

El sistema calcula automáticamente:

- **Reducción de distancia**: `(distancia_actual - distancia_nueva) / distancia_actual * 100`
- **Ahorro de tiempo**: Minutos ahorrados por día/semana/mes
- **Impacto en productividad**: Basado en estudios que correlacionan distancia con productividad
- **Satisfacción proyectada**: Mejora en calidad de vida del colaborador

## 🔒 Consideraciones de Privacidad

- Los datos personales de colaboradores se manejan con estricta confidencialidad
- Las direcciones se procesan de forma segura
- Los resultados se presentan de forma agregada cuando es apropiado
- Cumplimiento con regulaciones de protección de datos

## 📚 Base Teórica

Este proyecto se fundamenta en investigación académica sobre:

- **Impacto del commuting en productividad** (Xiao, Wu & Kim, 2021)
- **Algoritmos de matching y teoría de juegos** (Shapley & Scarf, 1974)
- **Geografía económica y spatial mismatch** (Immergluck, 1998)
- **Optimización de recursos humanos** mediante métodos cuantitativos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea tu Feature Branch (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push al Branch (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es parte de una tesis académica. Para uso comercial, por favor contactar al autor.

## 👤 Autor

**[Jorfan Vargas]**
- Pregrado en Ciencia de Datos
- Universidad Externado de Colombia
- Email: jorfan.vargas@est.uexternado.edu.co
---

**Nota**: Este proyecto fue desarrollado como parte de una tesis de maestría en Ciencia de Datos, con el objetivo de demostrar la aplicación práctica de algoritmos de optimización en problemas reales de recursos humanos y logística empresarial.
