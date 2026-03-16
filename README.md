# Proto-Locality Simulation

Clean simulation code base for a coordinate-free weighted-network model of bounded relational separability, prepared for the Physical Review E submission workflow.

## Overview

This repository contains the core numerical code used to simulate a relational weighted-network model with:

- similarity-gated reinforcement,
- uniform decay,
- information-capacity suppression.

The code is used to study the onset of clustered, weakly bridged structure in a coordinate-free setting.

## Repository contents

- `proto_locality_core.py` — main simulation script
- `make_structure_figure.py` — generates structural heatmaps from saved final weight matrices
- `default_v0.json` — example configuration file
- `requirements.txt` — minimal Python dependencies

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt