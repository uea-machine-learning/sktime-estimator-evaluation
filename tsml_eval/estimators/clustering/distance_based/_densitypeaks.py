from typing import Dict, Optional, Union
import numpy as np
from sklearn.cluster import DBSCAN
from aeon.clustering.base import BaseClusterer
from aeon.distances import pairwise_distance


class TimeSeriesDensityPeaks(BaseClusterer):
