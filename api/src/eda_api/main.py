"""Entry point:  uvicorn eda_api.main:app --reload   then open http://localhost:8000/docs"""

import logging

from eda_api.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
