import re
import unicodedata
from typing import List, Optional, Tuple, Iterable, Set
import pandas as pd

# =========================
# Utilitarios básicos
# =========================

def _strip_accents(text: str) -> str:
    if text is None:
        return text
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _casefold(text: str) -> str:
    return text.casefold() if hasattr(text, "casefold") else text.lower()

def _alpha_num_spacing(text: str) -> str:
    """
    Inserta espacios entre letras y números donde falten: 'cr7sur' -> 'cr 7 sur'
    y alrededor de '#', '-' cuando estén pegados.
    """
    if not text:
        return text
    # letra + número / número + letra
    text = re.sub(r'(?<=[A-Za-zÁÉÍÓÚáéíóúñÑ])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-zÁÉÍÓÚáéíóúñÑ])', ' ', text)
    # Separar pegados con '#', '-', '–', '—'
    text = re.sub(r'\s*#\s*', ' # ', text)
    text = re.sub(r'\s*[-–—]\s*', ' - ', text)
    return _normalize_spaces(text)

# =========================
# Diccionarios y patrones
# =========================
# Abreviaturas de tipo de vía y equivalencias comunes en CO
_ABBR_MAP = [
    (r'\b(?:cll|cl)\b\.?', 'calle'),
    (r'\b(?:crr|cr|cra|kra|kr)\b\.?', 'carrera'),
    (r'\b(?:av|avda|aven)\b\.?', 'avenida'),
    #(r'\b(?:ac)\b\.?', 'avenida calle'),      # Bogotá: AC 68
    #(r'\b(?:ak)\b\.?', 'avenida carrera'),    # Bogotá: AK 30
    (r'\b(?:dg|diag)\b\.?', 'diagonal'),
    (r'\b(?:tv|transv)\b\.?', 'transversal'),
    (r'\b(?:circ)\b\.?', 'circunvalar'),
    (r'\b(?:aut|autop|au)\b\.?', 'autopista'),
    (r'\b(?:km|kil)\b\.?', 'kilometro'),
]

_NUM_ALIAS = re.compile(r'\b(?:nº|n|n°|nº\.|no\.?|num\.?|nro\.?|numero)\b', re.I)

# Palabras que indican información posterior no útil para geocodificar
_SUFFIX_NO_GEOCODE = re.compile(
    r'\b(?:ap|apt|apto|apartamento|int|interior|to|torre|blq|bloque|bodega|'
    r'local|ofi|oficina|piso|nivel|edificio|adm|administraci[oó]n|agrupaci[oó]n|'
    r'agrupacion|barrio|mz|manzana|lt|lote|unidad|conjunto|casa|apartado|ph)\b.*$',
    re.I
)

# Tipo de vía esperado (para inferencias y validaciones)
_TIPO_VIA_RE = r'(?:calle|carrera|avenida(?:\s+(?:calle|carrera))?|autopista|diagonal|transversal|circular|kilometro)'

# =========================
# Funciones de normalización
# =========================

def _normalizar_abreviaturas(direccion: str) -> str:
    d = f" {direccion} "
    for pat, repl in _ABBR_MAP:
        d = re.sub(pat, repl, d, flags=re.I)
    # normalizar 'bis' separada
    d = re.sub(r'\bbis\b', ' bis ', d, flags=re.I)
    # normalizar orientaciones
    d = re.sub(r'\b(sur|norte|este|oeste)\b', r' \1 ', d, flags=re.I)
    return _normalize_spaces(d)

def _normalizar_separadores(direccion: str) -> str:
    d = direccion
    d = _NUM_ALIAS.sub('#', d)
    # Mantener # y - como separadores válidos; el resto de signos -> espacio
    d = d.replace('º', '').replace('ª', '').replace('°', '')
    d = re.sub(r'[^\w#\-\s/]', ' ', d, flags=re.I)  # conserva letras, dígitos, #, -, /
    d = _alpha_num_spacing(d)
    return d

def _inferir_separadores(direccion: str) -> str:
    """
    Si hay patrón 'tipo pri sec placa' => 'tipo pri # sec - placa'
    También agrega '-' después de '# sec' cuando falta.
    """
    d = direccion

    # 1) Inferir # y - cuando están los 3 números seguidos
    pat = re.compile(
        rf'^(?P<tipo>{_TIPO_VIA_RE})\s+'
        r'(?P<pri>\d+[a-z]?(?:\s+bis)?)\s+'
        r'(?:(?P<ori>sur|norte|este|oeste)\s+)?'
        r'(?P<sec>\d+[a-z]?)\s+'
        r'(?P<pla>\d+)\b',
        re.I
    )
    def _sub1(m: re.Match) -> str:
        tipo = _normalize_spaces(m.group('tipo'))
        pri = _normalize_spaces(m.group('pri'))
        ori = m.group('ori')
        sec = m.group('sec')
        pla = m.group('pla')
        core = f"{tipo} {pri}"
        if ori:
            core += f" {ori}"
        return f"{core} # {sec} - {pla}"

    d = pat.sub(_sub1, d)

    # 2) Si hay '# sec placa' sin '-', agrégalo
    d = re.sub(r'(#\s*\d+[a-z]?)\s+(?=\d+\b)', r'\1 - ', d, flags=re.I)

    return _normalize_spaces(d)

def _remover_sufijo_no_geocode(direccion: str) -> str:
    return _normalize_spaces(_SUFFIX_NO_GEOCODE.sub('', direccion))

def _eliminar_cero_si_sigue_numero(direccion: str) -> str:
    # 'calle 0 9' -> 'calle 9' ; 'calle 09' -> 'calle 9'
    return re.sub(r'\b0\s*(?=\d)', '', direccion)

def _to_canon(direccion: str) -> str:
    """
    Secuencia de limpieza completa para una sola dirección.
    Si la primera palabra es una palabra clave a eliminar y la segunda es una palabra común de dirección,
    elimina solo la primera palabra antes de continuar con la limpieza.
    """
    if not isinstance(direccion, str) or not direccion.strip():
        return direccion
    d = direccion.strip()
    palabras = d.split()
    if palabras:
        primera = palabras[0].lower()
        segunda = palabras[1].lower() if len(palabras) > 1 else ""
        if primera in palabras_clave and segunda in _palabras_comunes_direccion:
            d = " ".join(palabras[1:])
    d = _alpha_num_spacing(d)                   # separación básica
    d = _normalizar_abreviaturas(d)            # cl->calle, cra->carrera, ak/ac...
    d = _normalizar_separadores(d)             # No/N° -> # ; quitar signos raros
    d = _eliminar_cero_si_sigue_numero(d)      # 0 antes de dígitos
    d = _inferir_separadores(d)                # inferir # y -
    d = _remover_sufijo_no_geocode(d)          # cortar apto/interior/etc.
    d = _normalize_spaces(d)
    return d

# =========================
# API pública (compatibilidad + nuevas)
# =========================

def eliminar_despues(direccion: str, palabras: Iterable[str]) -> str:
    """
    Elimina todo lo que está DESPUÉS de cualquiera de las palabras dadas (como sufijo).
    Usa límites de palabra para evitar cortes indeseados (p.ej. 'ap' dentro de otra palabra).
    """
    if pd.isna(direccion):
        return direccion
    d = direccion
    for p in palabras:
        if not p:
            continue
        d = re.sub(rf'\b{re.escape(p)}\b.*$', '', d, flags=re.I)
    return _normalize_spaces(d)

# Alias para compatibilidad con tu versión original con typo
eliminiar_despues = eliminar_despues

def reemplazar(direccion: Optional[str]) -> Optional[str]:
    """
    Versión robusta: normaliza abreviaturas, separadores, orientaciones, BIS y ruido.
    (Mantiene el nombre original de tu módulo)
    """
    if pd.isna(direccion):
        return direccion
    return _to_canon(direccion)

# Palabras clave para eliminar info adicional 
palabras_clave = [
    'casa', 'ap', 'apt', 'apto', 'apartamento', 'int', 'interior',
    'to', 'torre', 'bloque', 'bodega', 'local', 'oficina', 'piso',
    'nivel', 'edificio', 'administracion', 'agrupacion', 'mz', 'manzana',
    'lt', 'lote', 'unidad', 'conjunto', 'ph', 'barrio', 'apartado'
]

# Palabras comunes de dirección
_palabras_comunes_direccion = {
    "calle", "carrera", "avenida", "autopista", "diagonal", "transversal", "circular", "kilometro"
}

def eliminar_info_adicional(direccion: Optional[str]) -> Optional[str]:
    """
    Corta a partir de términos que añaden ruido para geocodificar.
    Si la palabra clave a eliminar está en la primera posición y la segunda es una palabra común de dirección,
    elimina solo la primera palabra. Si la palabra clave aparece en otra posición, elimina desde esa palabra en adelante.
    """
    if pd.isna(direccion):
        return direccion
    d = _to_canon(direccion)
    palabras = d.split()
    if palabras:
        primera = palabras[0].lower()
        segunda = palabras[1].lower() if len(palabras) > 1 else ""
        if primera in palabras_clave and segunda in _palabras_comunes_direccion:
            # Elimina solo la primera palabra
            d = " ".join(palabras[1:])
            return _normalize_spaces(d)
    # Si no cumple la condición, elimina desde la palabra clave en adelante
    d = _remover_sufijo_no_geocode(d)
    return _normalize_spaces(d)

def eliminar_repeticion_final(cadena: str) -> str:
    if not isinstance(cadena, str):
        return cadena
    palabras = cadena.strip().split()
    if not palabras:
        return cadena
    ultima = palabras[-1]
    sin_rep = [p for p in palabras[:-1] if p != ultima]
    sin_rep.append(ultima)
    return ' '.join(sin_rep)

def _norm_ciudad(texto: str) -> str:
    if not isinstance(texto, str):
        return ''
    return _normalize_spaces(_strip_accents(_casefold(texto)))

def ult_palabra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mejora: compara dirección y ciudad sin tildes y casefold.
    Si la ciudad ya aparece al final de la dirección, no duplica.
    """
    dir_col = '079_DIRECCION'
    ciudad_col = 'CIUDAD RESIDENCIA'
    if dir_col not in df.columns or ciudad_col not in df.columns:
        return df

    for i in df.index:
        direccion = str(df.at[i, dir_col]) if pd.notna(df.at[i, dir_col]) else ''
        ciudad = str(df.at[i, ciudad_col]) if pd.notna(df.at[i, ciudad_col]) else ''
        direccion = eliminar_repeticion_final(direccion)
        if not direccion or not ciudad:
            continue
        last_dir = _norm_ciudad(direccion).split()[-1] if direccion.strip() else ''
        last_city = _norm_ciudad(ciudad).split()[-1] if ciudad.strip() else ''
        if last_dir != last_city:
            df.at[i, dir_col] = f"{direccion}, {ciudad}"
    return df

def eliminar_cero_si_sigue_numero(direccion: Optional[str]) -> Optional[str]:
    if pd.isna(direccion):
        return direccion
    return _eliminar_cero_si_sigue_numero(str(direccion))

# =========================
# Funciones nuevas para geocodificación
# =========================
def normalizar_direccion(direccion: Optional[str]) -> Optional[str]:
    """
    Normalización canónica pensada para enviar al geocoder (sin sufijos).
    """
    if pd.isna(direccion):
        return direccion
    return _to_canon(str(direccion))

def anexar_contexto(direccion: str,
                    ciudad: Optional[str] = None,
                    departamento: Optional[str] = None,
                    pais: Optional[str] = "Colombia") -> str:
    """
    Anexa ciudad, departamento y país para ayudar al geocoder a desambiguar.
    Evita duplicados simples.
    """
    partes = [direccion]
    base_norm = _norm_ciudad(direccion)

    # Fuerza país a "Colombia" siempre
    pais = "Colombia"

    for comp in (ciudad, departamento, pais):
        if comp and _norm_ciudad(comp) not in base_norm:
            partes.append(str(comp))
            base_norm += " " + _norm_ciudad(comp)
    return _normalize_spaces(', '.join(partes))

def generar_variantes_geocoding(direccion: str) -> List[str]:
    """
    Genera variantes útiles (no explosivas) para un mismo address.
    - Con y sin abreviaturas AK/AC.
    - Con y sin 'sur'.
    - Con y sin '#'.
    """
    variants: Set[str] = set()
    d = direccion

    # Base
    variants.add(d)

    # Con/ sin 'sur'
    if re.search(r'\bsur\b', d, re.I):
        variants.add(_normalize_spaces(re.sub(r'\bsur\b', '', d, flags=re.I)))
    else:
        # (No añadimos 'sur' si no estaba, para evitar falsos)
        pass

    # Con y sin '#'
    if '#' in d:
        variants.add(_normalize_spaces(d.replace('#', '')))
    else:
        # Intento prudente: insertar '#' si patrón permite
        d2 = re.sub(rf'^({_TIPO_VIA_RE}\s+\d+[a-z]?(?:\s+bis)?(?:\s+(?:sur|norte|este|oeste))?)\s+(\d+[a-z]?)\s*-\s*(\d+)\b',
                    r'\1 # \2 - \3', d, flags=re.I)
        if d2 != d:
            variants.add(d2)

    # AK / AC alternos
    d_ak = re.sub(r'\bavenida carrera\b', 'AK', d, flags=re.I)
    d_ac = re.sub(r'\bavenida calle\b', 'AC', d, flags=re.I)
    variants.update({_normalize_spaces(d_ak), _normalize_spaces(d_ac)})

    # Abreviaturas de calle/carrera
    variants.add(re.sub(r'\bcalle\b', 'CL', d, flags=re.I))
    variants.add(re.sub(r'\bcarrera\b', 'CR', d, flags=re.I))

    # Filtrar vacíos y normalizar
    variants = {_normalize_spaces(v) for v in variants if v and v.strip()}
    # Orden heurístico: preferir forma canónica con '#'
    ordered = sorted(variants, key=lambda x: (0 if '#' in x else 1, len(x)))
    return ordered[:6]  # límite prudente

def preparar_para_geocoding(
    df: pd.DataFrame,
    col_dir: str,
    col_ciudad: Optional[str] = None,
    col_departamento: Optional[str] = None,
    col_pais: Optional[str] = None,
    salida_col_base: str = "direccion_canon",
    salida_col_geocode: str = "direccion_geocode",
    salida_col_variantes: str = "direccion_variantes"
) -> pd.DataFrame:
    """
    Pipeline vectorizado:
    1) Normaliza la dirección.
    2) Elimina sufijos que confunden.
    3) Anexa ciudad/depto/país si están disponibles.
    4) Genera variantes.

    Devuelve el DF con 3 columnas nuevas:
      - salida_col_base: dirección normalizada sin contexto
      - salida_col_geocode: con contexto (ciudad/depto/país)
      - salida_col_variantes: lista de variantes (hasta 6)
    """
    if col_dir not in df.columns:
        raise KeyError(f"No existe la columna '{col_dir}' en el DataFrame.")

    # 1) Normalizar base
    df[salida_col_base] = df[col_dir].apply(normalizar_direccion)

    # 2) Con contexto
    def _ctx(row) -> str:
        # Fuerza país a "Colombia" siempre
        return anexar_contexto(
            row.get(salida_col_base, ''),
            row.get(col_ciudad) if col_ciudad in df.columns else None,
            row.get(col_departamento) if col_departamento in df.columns else None,
            "Colombia"
        )

    df[salida_col_geocode] = df.apply(_ctx, axis=1)

    # 3) Variantes
    df[salida_col_variantes] = df[salida_col_geocode].apply(generar_variantes_geocoding)

    return df

