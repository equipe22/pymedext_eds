# -*- coding: utf-8 -*-
from unidecode import unidecode
from collections import OrderedDict
import re
from collections import Counter


from .constants import NUMBER, FREQ_REPLACE, FREQ_EXP, NUMBER_4_DOSE, UNITS


#####DRUG NORMALIZATION
def clean_drug(entity, romedi): 
    for field in ['BN_label', 'PIN_label', 'IN_label', 'IN_hidden_label', 'BN_hidden_label']:
        entity = unidecode(entity)
        if field in ['IN_hidden_label', 'BN_hidden_label']: 
            str_to_match =  entity.lower()
        else: 
            str_to_match =  entity.upper()

        match = [x for x in romedi.infos if  str_to_match in x[field]]

        if len(match)> 0: 
            res = match[0]
            res = { k: res[k] for k in ['BN_label', 'PIN_label', 'IN_label', 'ATC7','ATC5','ATC4'] }
            break     
            
        else: 
            res = {}
    return res



def clean_class(entity, class_norm): 
    
    res = [v[0] for k,v in class_norm.items() if entity.upper() == k]
    if res != []:
        return res[0]
    else: 
        return []




def norm_freq(string):
    unfolded_freq = [t for sub in [[(k, u) for u in v] for k, v in FREQ_EXP.items()] for t in sub]
    freq = None
    for key, pattern in unfolded_freq:
        if re.search(pattern, string):
            freq = key
            break
    return freq

def clean_freq(string):
    """
    Normalize frequencies
    Daily information is expressed in H_1 1 1 1 format to approximate (8h 12h 16h 22h)
    Imputation is done as the following:
        - one per day is 1 0 0 0
        - 2 per day is 1 0 0 1
        - 3 per day is 1 1 0 1
    Otherwise information is expressed as a frequency (J_1/2 is every two day, S_1/3 is every three weeks)
    
    """
    #clean
    string = unidecode(string)
    string = re.sub("[^A-Za-z0-9]", " ", string)
    string = string.strip().lower()
    
    #replace number
    for k, v in NUMBER.items():
        string = re.sub(k, v, string)
    #replace stop words
    for k, v in FREQ_REPLACE.items():
        string = re.sub(k, v, string)

    string = re.sub(' +', ' ', string)

    #replace exp
    freq = norm_freq(string)


    return freq

##################################################################################
# ###########################FDOSE NORMALISATION  ################################
# #################################################################################

unfolded_units = [t for sub in [[(k, u) for u in v] for k, v in UNITS.items()] for t in sub]
dose_regex = re.compile("(\d ?x ?)?(\d+[,\.]?\d{0,4})(x\d)?\s*("+ "|".join([t[1] for t in unfolded_units]) + ")?(x\d)?")


def norm_value(value, modifier):
    try:
        value = value.replace(",", ".")
        value = float(value)
    except:
        value = -1

    for mod in modifier:
        if mod != "":
            break
    try:
        mod = int(re.search("\d+", mod).group(0))
    except:
        mod = 1
    return value * mod

def unidecode_not(string, notunidecode="µμ"):
    acc = ""
    for c in string:
        if c not in notunidecode:
            acc += unidecode(c)
        else:
            acc += c
    return acc

def norm_unit(unit):
    if unit == "":
        unit = None
    else:
        for key, pattern in unfolded_units:
            if re.match(pattern, unit):
                unit = key
                break
    return unit

def clean_dose(string):
    """
    Normalise the dose
    Returns a list of tuple (dose, unit)
    Unit is unit if not specific, NA if no unit found
    """
    #clean
    string = unidecode_not(string)
    string = re.sub("[^A-Za-z0-9,\.%µμ/]", " ", string)
    string = re.sub("(\d) ([\.,/]\d)", '\g<1>\g<2>', string) #tokenized 3 .5mg 1 /2

    string = string.strip().lower()
    
    #replace number
    for k, v in NUMBER_4_DOSE.items():
        string = re.sub(k, v, string)
    #tokenization issue to fix using raw mention to be tokenization independent
    string = re.sub("(\d+) ?i?eme", '1', string) #tokenized 3eme
    string = re.sub("(1\/2)","0.5", string) #1/2
    string = re.sub("(1 2)","0.5", string) #1/2
    string = re.sub("(1\/3)","0.33", string) #1/2
    string = re.sub("(2\/3)","0.66", string) #1/2
    string = re.sub("(1\/4)","0.25", string) #1/2
    string = re.sub(" 000","000", string) #1 000 000


    #iter on matched patterns of type (dose, unit)
    normed_dose = []
    for modifier0, value, modifier1, unit, modifier2 in dose_regex.findall(string):

        value = norm_value(value, (modifier0, modifier1, modifier2))
        unit = norm_unit(unit)
        normed_dose.append("val:{}__{}".format(value, unit))    

    return "|".join(normed_dose)
