#!/usr/bin/env python3
"""Return a list of ships"""


import requests


def availableShips(passengerCount):
    """Return a list of ships"""
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships'
    while url is not None:
        data = requests.get(url).json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            if passengers == 'n/a' or passengers == 'unknown':
                passengers = -1
            if int(passengers) >= passengerCount:
                ships.append(ship['name'])
        url = data['next']
    return ships
