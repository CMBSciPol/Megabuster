import logging
import os

import megabuster.compsep 
import megabuster.io 
import megabuster.precomputations 
import megabuster.tools 
import megabuster.helper



logger = logging.getLogger("megabuster")

if not logging.getLogger().handlers:

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())