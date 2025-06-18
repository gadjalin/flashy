from enum import IntEnum


# eosData indices
class EOS_VAR(IntEnum):
    # Primary variables
    PRES = 0
    DENS = 1
    EINT = 2
    TEMP = 3
    GAMC = 4
    ABAR = 5
    ZBAR = 6
    ENTR = 7
    EKIN = 8

    # Derivatives
    DPT = 9
    DPD = 10
    DET = 11
    DED = 12
    DEA = 13
    DEZ = 14
    DST = 15
    DSD = 16
    CV  = 17
    CP  = 18
    PEL = 19
    NE  = 20
    ETA = 21

    NUM = 22

    # Extra variables for analysis
    PRAD  = 0
    PION  = 1
    PELE  = 2
    PCOUL = 3

    ERAD  = 4
    EION  = 5
    EELE  = 6
    ECOUL = 7

    SRAD  = 8
    SION  = 9
    SELE  = 10
    SCOUL = 11

    NUM_EXTRA = 12
