import numpy as np

error = lambda label, out : np.sum( ( label - out )**2 )
