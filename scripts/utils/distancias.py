import googlemaps

api_key = 123456789

# Cliente de googlemaps
gmaps = googlemaps.Client(key=api_key)

def distancia_duracion_car(origin, destination):
    try:
        result = gmaps.distance_matrix(origin, destination, mode="driving")
        element = result['rows'][0]['elements'][0]
        if element['status'] == 'OK':
            dist_km = element['distance']['value'] / 1000  # metros a km
            dur_min = element['duration']['value'] / 60    # segundos a minutos
            return dist_km, dur_min
    except Exception as e:
        print(f"Error en API: {e}")
    return None, None

def distancia_duracion_transporte(origin, destination):
    try:
        result = gmaps.distance_matrix(origin, destination, mode="transit")
        element = result['rows'][0]['elements'][0]
        if element['status'] == 'OK':
            dist_km = element['distance']['value'] / 1000  # metros a km
            dur_min = element['duration']['value'] / 60    # segundos a minutos
            return dist_km, dur_min
    except Exception as e:
        print(f"Error en API: {e}")
    return None, None