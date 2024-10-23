import os

CDIR = os.path.dirname(os.path.realpath(__file__))
WORKDIR = os.path.abspath(os.path.join(CDIR, '..'))
DATADIR = os.path.abspath(os.path.join(WORKDIR, "data"))
EXPSDIR = os.path.abspath(os.path.join(WORKDIR, "experiments"))
PODATADIR = os.path.abspath(os.path.join(DATADIR, "private_obfuscation"))
LOCAL_DATADIR = os.path.abspath(os.path.join(CDIR, "local_data"))
os.makedirs(DATADIR, exist_ok=True)


