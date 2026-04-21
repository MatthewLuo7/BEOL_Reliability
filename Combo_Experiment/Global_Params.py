V_OP = 0.7  # V, operating voltage

PITCH = 21.0  # nm

SPACING = 9.0  # nm

T_TARGET = 10* 365.25 * 24 * 3600  # 10-year lifetime in seconds

F_TARGET = 100/1000000  # 100 ppm allowed failure rate

N_VIA = 2e9        # Total number of vias

N_TIP = 2e9        # Total number of tips

LINE_WIDTH = PITCH - SPACING  # nm

LINE_HEIGHT = LINE_WIDTH * 2  # nm (Aspect Ratio = 2x) (given in paper 2)

TOTAL_LINE_LENGTH_NM = 3 * 1e12  # 3 km in nm

TOTAL_AREA_NM2 = TOTAL_LINE_LENGTH_NM * LINE_HEIGHT  # Total area A in nm^2

SIGMA_DIE = 0.2  # nm, die-to-die spacing variation

RHO_LER = 0.8  # LER cross-correlation for multi-patterning