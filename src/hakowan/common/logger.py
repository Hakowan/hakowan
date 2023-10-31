import logging
import lagrange

logger = logging.getLogger("hakowan")
logger.addHandler(lagrange.logger.handlers[0])
