from unidecode import unidecode
from collections import OrderedDict
import re
from collections import Counter


#####DRUG NORMALIZATION
def clean_drug(entity, romedi): 
    for field in ['BN_label', 'PIN_label', 'IN_label', 'IN_hidden_label', 'BN_hidden_label']:
        if field in ['IN_hidden_label', 'BN_hidden_label']: 
            str_to_match =  entity.lower()
        else: 
            str_to_match =  entity.upper()

        match = [x for x in romedi.infos if  str_to_match in x[field]]

        if len(match)> 0: 
            res = match[0]
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


###FREQ DOSE NORMALIZATION

# def fusion_norm_freq(drugblob):
#     if drugblob["ENT/FREQ"]:
#         freq_mention = [t["mention"] for t in drugblob["ENT/FREQ"]]
#         freq = [ clean_freq(t)[0] for t in freq_mention]
#     else:
#         freq = []
#         freq_mention = []

#     if drugblob["ENT/DOSE"]:
#         dose_mention = [t["mention"] for t in drugblob["ENT/DOSE"]]
#         dose = [ clean_dose(t)[0] for t in dose_mention]
#     else:
#         dose = []
#         dose_mention = []
        

#     blob_norm = []
#     for freq, (value, unit) in zip(freq, [t for sub in dose for t in sub]):
#         blob_norm.append((value, unit, freq))
        
        
#     return blob_norm, freq_mention, dose_mention

number = {'zero': '0',
 'une ': '1 ',
 'un ': '1 ',

 'deux': '2',
 'trois': '3',
 'quatre': '4',
 'cinq': '5',
 'six': '6',
 'sept': '7',
 'huit': '8',
 'neuf': '9',
 'dix': '10',
 'onze': '11',
 'douze': '12',
 'treize': '13',
 'quatorze': '14',
 'quinze': '15',
 'seize': '16',
 'dix-sept': '17',
 'dix-huit': '18',
 'dix-neuf': '19'}

freq_replace = {"les ":" ",
                "le ":" ",
                "par ":" ",
                " et ":" ",
                "a ":" ",
                "fois ":" ",
                "x ":" ",
                "^jr?$":"jour",
                "jours":"jour",
                r'(\d) ?jr?$': r'\g<1> jour',
                '(\d) ?h(\s|$)':'\g<1> heure\g<2>',
                r'x ?(\d)': r'\g<1>',
                "heures":"heure",
                "semaines":"semaine",

                "toutes|tous|toute":"tout"
               }

freq_exp = OrderedDict({
    "error":["0 midi", "1 .0.0", "garde veine"],
    "J_1/2":["1 jour sur 2", "tout 2 jour"],

    "H_1 1 1 1":["3 4 jour", "4 jour", "tout 6 heure", "6 heure", "^1 1 1 1$"],
    "H_1 1 0 1":["matin midi soir", "3 jour", "tout 8 heure", "8 heure", "^1 1 1$"],
    "H_1 0 0 1":["(1 )?matin soir", "2 jour", "^1 0 1$"],
    "H_1 1 0 0":["matin midi", "^1 1 0$"],
    "H_0 1 0 1":["midi soir", "^0 0 1$"],
    "H_0 0 0 1":["(1 )?soir",  "au coucher", "20 00", "^0 0 1$"],
    "H_0 1 0 0":["midi", "^0 1 0$"],
    "H_0 0 1 0":["au gouter"],

    "H_2 2 2 2":["^2 2 2 2$"],
    "H_2 2 0 2":["^2 2 2$"],
    "H_2 0 0 2":[ "^2 0 2$"],
    "H_2 2 0 0":[ "^2 2 0$"],
    "H_0 0 0 2":[ "^0 0 2$"],
    "H_0 2 0 0":[ "^0 2 0$"],
    "H_2 0 0 0":[ "^2 0 0$"],
    "H_3 3 3 3":["^3 3 3 3$"],
    "H_3 3 0 3":["^3 3 3$"],
    "H_3 0 0 3":[ "^3 0 3$"],
    "H_3 3 0 0":[ "^3 3 0$"],
    "H_0 0 0 3":[ "^0 0 3$"],
    "H_0 3 0 0":[ "^0 3 0$"],
    "H_3 0 0 0":[ "^3 0 0$"],


    #semaine
    "S_1/2":["tout 2 semaine", "tout 15 jour"],
    "S_1/3":["tout 3 semaine"],
    "S_1/6":["tout 6 semaine"],
    "S_1/8":["tout 2 mois", "2 mois", "tout 8 semaine"],
    "S_1/13":["tout 3 mois", "3 mois", "tout 12 semaine", "trimestrielle"],
    "S_1/26":["tout 6 mois", "6 mois", "semestrielle"],
    "S_1/17":["tout 4 mois", "4 mois"],
    
    #jour
    "J_3/7":["3 semaine"],

    #unspecifix, FP generation
    "S_1/4":["tout mois", "mois", "tout 4 semaine", "mensuelle"],
    "J_1/7":["(1 )?semaine", "hebdomadaire", " lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"],
    "H_1 0 0 0":["(1 )?matin", "tout jour", "(1 )?jour", "tout 24 heure", "24 heure", "09 00",  "^1 0 0$"],

}
)

unfolded_freq = [t for sub in [[(k, u) for u in v] for k, v in freq_exp.items()] for t in sub]


def norm_freq(string):
    freq = "NA"
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
    for k, v in number.items():
        string = re.sub(k, v, string)
    #replace stop words
    for k, v in freq_replace.items():
        string = re.sub(k, v, string)

    string = re.sub(' +', ' ', string)

    #replace exp
    freq = norm_freq(string)


    return freq

##################################################################################
############################FDOSE NORMALISATION  ################################
##################################################################################

number_4_dose = {'zero': '0',
 'une ': '1 ',
 'un ': '1 ',
 'deux': '2',
 'trois': '3',
 'quatre': '4',
 'cinq': '5',
 'six': '6',
 'sept': '7',
 'huit': '8',
 'neuf': '9',
 'dix': '10',
 'onze': '11',
 'douze': '12',
 'treize': '13',
 'quatorze': '14',
 'quinze': '15',
 'seize': '16',
 'dix-sept': '17',
 'dix-huit': '18',
 'dix-neuf': '19',
 #autre
"fois ": "x ",
"double":"2",
"triple": "3"
         }

units = OrderedDict({
    "mg/kg":["mg kg"],
    "mg/ml":["mg ml"],
    "µg/kg":["[µμ]g kg", "mcg kg"],
    "µg/ml":["[µμ]g ml", "mcg ml"],

    "unit/kg":["goutte par kg"],
    "%":["%", "pourcent"],
    "mg":["mg", "milligrammes?"],
    "ml":["ml", "millilitres?"],
    "unit":["cps?", "comprimes?", "cpr",
          "ampoules?", "amp",
            "doses?",
          "injections?", 
          "applications?", "patchs?", "tubes?",
          "gell?ule?",
          "bouffees?", 
          "gouttes?", "gttes?", 
          "flacon",
          "sachet",
          "ui", "unites?",
          "pulverisations?", "inhalations?",
          "perfusions?", "bolus",
            "cycle", "cure"
           ],
    "g":["g", "grammes?"],
    "µg":["[µμ]g", "microgrammes?", "mcg"],
    "µmol":["[µμ]mol", "micromol", "mcmol"],
    "kcal":["kcal", "kilocalories?"],
    "cc":["cc", "centilitre"]
        })

unfolded_units = [t for sub in [[(k, u) for u in v] for k, v in units.items()] for t in sub]
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
        unit = "NA"
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
    for k, v in number_4_dose.items():
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