#!/usr/bin/env python3
"""Return a list of ships"""


import requests


def sentientPlanets():
    """Return a list of ships"""
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species'
    while url is not None:
        data = requests.get(url).json()
        for species in data['results']:
            if ((species['designation'] == 'sentient'
                 or species['designation'] == 'reptilian')):
                if species['homeworld'] is not None:
                    hw = requests.get(species['homeworld']).json()
                    planets.append(hw['name'])
        url = data['next']
    return planets
