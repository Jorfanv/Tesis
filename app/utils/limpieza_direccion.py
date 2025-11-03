def eliminiar_despues (direccion, palabras):
    '''
    Elimina todo lo que está después de una palabra específica en una dirección.
    
    Parámetros:
    direccion (str): La dirección original.
    palabras (list): Lista de palabras clave
    '''

    for palabra in palabras:
        direccion = direccion.split(palabra)[0]
    return direccion