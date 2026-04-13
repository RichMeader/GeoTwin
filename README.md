# GeoTwin

Open digital twin simulation platform for geothermal drilling.

## Overview
GeoTwin is a physics-first geothermal drilling simulator built for hard-rock 
formations. It models Rate of Penetration, Mechanical Specific Energy, bit wear, 
mud hydraulics, and thermal profiles for crystalline rock environments.

## Target environments
- United Downs, Cornwall (Carnmenellis Granite)
- Utah FORGE, Milford Utah (granitoid basement) — validated against open DOE data

## Licence
MIT — open source, free to use and build upon.

## Activating the environment
source venv/bin/activate
python geotwin_engine/geotwin_engine.py

## Using GeoTwin
1. Clone the repo
git clone https://github.com/RichMeader/GeoTwin.git
cd GeoTwin

4. Create a virtual environment
python3 -m venv venv
source venv/bin/activate (Mac/Linux, or on Windows: venv\Scripts\activate)

5. Install dependencies
pip install numpy scipy matplotlib pandas pyyaml

6. Run the simulator
python geotwin_engine/geotwin_engine.py
