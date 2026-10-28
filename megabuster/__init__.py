import logging
import os

from megabuster.compsep import *
from megabuster.io import *
from megabuster.precomputations import *
from megabuster.tools import *
from megabuster.helper import *



logger = logging.getLogger("megabuster")

if not logging.getLogger().handlers:

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())