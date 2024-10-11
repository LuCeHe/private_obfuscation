import os

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))
DATADIR = os.path.abspath(os.path.join(WORKDIR, "data"))
PODATADIR = os.path.abspath(os.path.join(DATADIR, "private_obfuscation"))
os.makedirs(DATADIR, exist_ok=True)
