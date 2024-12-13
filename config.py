# Configuration settings
DYSLEXIA_DATA_PATH = 'dyslexia_data/dyslexic/*.csv'
NON_DYSLEXIA_DATA_PATH = 'dyslexia_data/non_dyslexic/*.csv'

# Model parameters
SEQUENCE_LENGTH = 200
N_FEATURES = 4

# Eye tracking parameters
FIXATION_THRESHOLD = 20  # milliseconds
SACCADE_VELOCITY_THRESHOLD = 0.01  # units per millisecond
PATTERN_WINDOW_SIZE = 500  # milliseconds 