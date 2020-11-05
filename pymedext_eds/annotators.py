from pymedextcore.annotators import Annotation, Annotator
from pymedextcore.document import Document
import glob
import re
import json
from flashtext import KeywordProcessor
import pandas as pd

try:
    from quickumls import QuickUMLS
    from quickumls.constants import ACCEPTED_SEMTYPES
except: 
    print('QuickUMLS not installed. Please use "pip install quickumls"')

    
from .constants import SECTION_DICT   
from .verbs import verbs_list

def rawtext_loader(file): 
    with open(file) as f:
        txt = f.read()
        ID = re.search("([0-9]+)\.txt$", file)
        if not ID:
            ID = file
        else:
            ID  = ID.groups()[0]
    return Document(
        raw_text = txt,
        ID = ID
    )


class Endlines(Annotator):
    
    def annotate_function(self, _input):
        inp= self.get_first_key_input(_input)[0]
        txt = self.preprocess(inp.value)
        txt = self.gerer_saut_ligne(txt)
        txt = self.gerer_parenthese(txt)

        txt = re.sub(r"CONCLUSIONS?([A-Za-z])",r"CONCLUSION \1", txt,re.IGNORECASE)
        
        phrases = re.split(r'([\r\n;\?!.])', txt)
        
        return([Annotation(
            type = self.key_output,
            value = txt, 
            span = (0, len(txt)),
            source = self.ID,
            source_ID = inp.ID
        )])
    
    @staticmethod
    def preprocess(txt):
        txt = re.sub("\r"," ", txt)
        txt = re.sub("\[pic\]"," ", txt)
        return txt

    @staticmethod
    def gerer_saut_ligne(txt):
        txt=re.sub(r"M\.","M ",txt)
        txt=re.sub(r"Mme\.","Mme ",txt)
        txt=re.sub(r"Mlle\.","Mlle ",txt)
        txt=re.sub(r"Mr\.","Mr ",txt)
        txt=re.sub(r"Pr\.","Pr ",txt)
        txt=re.sub(r"Dr\.","Dr ",txt)
        txt=re.sub(r"([A-Z])\.([A-Z])",r"\1 \2",txt)
        txt=re.sub(r"([0-9])\.([0-9])",r"\1,\2",txt)
        txt=re.sub(r"(:\s*[a-z]+)\s*\n",r"\1.\n", txt, re.IGNORECASE)
        txt=re.sub(r"\+\n",r"+.\n", txt, re.IGNORECASE)

        # on nettoie les espaces multiples 
        txt = re.sub(r"([A-Za-z0-9,:])\s+([a-z0-9])",r"\1 \2",txt)  #modification nico 12 04 2014, sensible à la casse !!
        txt = re.sub(r"([A-Za-z0-9,:])\n *([a-z0-9])",r"\1 \2",txt)  # modification nico 12 04 2014, sensible à la casse !!
        txt=re.sub(r"\n\s*([^ ])",r"\n\1", txt)
        txt=re.sub(r"\n\s*([a-zA-Z0-9]+)\.?\n",r" \1.", txt)
        txt=re.sub(r"\n\s*([a-z0-9])",r" \1", txt)

        txt=re.sub(r"\n",". ", txt)

        txt=re.sub(r" de\s*\. "," de ", txt)
        txt=re.sub(r" par\s*\. "," par ", txt)
        txt=re.sub(r" le\s*\. "," le ", txt)
        txt=re.sub(r" le\s*\. "," le ", txt)
        txt=re.sub(r" du\s*\. "," du ", txt)
        txt=re.sub(r" la\s*\. "," la ", txt)
        txt=re.sub(r" les\s*\. "," les ", txt)
        txt=re.sub(r" des\s*\. "," des ", txt)
        txt=re.sub(r" un\s*\. "," un ", txt)
        txt=re.sub(r" une\s*\. "," une ", txt)
        txt=re.sub(r" ou\s*. "," ou ", txt) 

        txt=re.sub(r" pour *\. "," pour ", txt) 
        txt=re.sub(r" avec *\. "," avec ", txt) 

        txt=re.sub(r" \.\s*pour "," pour ", txt) 
        txt=re.sub(r" \.\s*avec "," avec ", txt) 
        txt=re.sub(r" \.\s*et "," et ", txt) 
        txt=re.sub(r" \.\s*avec "," avec ", txt) 
        return txt

    @staticmethod
    def gerer_parenthese(txt):
        txt=re.sub(r"\(-\)"," negatif ",txt)
        txt=re.sub(r"\(\+\)"," positif ",txt)

        # si plus de 30 caractères on met le contenu de la parenthese à la fin de la phrase
        txt=re.sub(r"\(([^)(]{30,5000})\)([^.]*)([\.])",r" \2 ; \1 \3",txt)

        # si entre 1 et 30 caractères on met des virgules  à la place des parentheses
        txt=re.sub(r"\(([^)(]{1,29})\)",r",\1 , ",txt)
        return txt

    
    @staticmethod
    def gerer_parenthese(txt):
        txt=re.sub(r"\(-\)"," negatif ",txt)
        txt=re.sub(r"\(\+\)"," positif ",txt)

        # si plus de 30 caractères on met le contenu de la parenthese à la fin de la phrase
        txt=re.sub(r"\(([^)(]{30,5000})\)([^.]*)([\.])",r" \2 ; \1 \3",txt)

        # si entre 1 et 30 caractères on met des virgules  à la place des parentheses
        txt=re.sub(r"\(([^)(]{1,29})\)",r",\1 , ",txt)
        return txt

    
class SentenceTokenizer(Annotator):
    
    def annotate_function(self, _input):
        inps= self.get_all_key_input(_input)
        
        res = []
        offset = 0
        
        for inp in inps:
        
            for sent in re.split(r'([\r\n;\?!.])', inp.value):
                if sent in ['.','', ' ', ';']:
                    continue

                start = inp.value.find(sent) + offset
                end = start + len(sent)

                res.append(Annotation(
                    type = self.key_output,
                    value = sent, 
                    span = (start, end),
                    source = self.ID,
                    source_ID = inp.ID
                ))
            offset = end
        return res


class Hypothesis(Annotator):
    def __init__(self, key_input, key_output, ID): 
        
        super().__init__(key_input, key_output, ID)
        self.verbs_hypo, self.verbs_all = self.load_verbs()
        
    def load_verbs(self): 
        #with open(json_file) as f:
        #    json_verbs = json.load(f)

        verbs_hypo = []
        verbs_all = []
        for verb in verbs_list:
            verbs_hypo += verb['conditionnel']['présent']
            verbs_hypo += verb['indicatif']['futur simple']

            for _, temps in verb.items():
                for _,v in temps.items():
                    verbs_all += v

        verbs_hypo = set(verbs_hypo)
        verbs_all = set(verbs_all)

        return verbs_hypo, verbs_all

    
    def detect_hypothesis(self, phrase):
            
        if len(phrase) > 150:
            return 'certain'
        else:
            phrase_low = phrase.lower()
            sentence_array = re.split(r'[^\w0-9.\']+', phrase_low)

            inter_hypo = list(set(self.verbs_hypo) & set(sentence_array))                

            if len(inter_hypo) > 0 :
                return 'hypothesis'
            else: 
                if ((re.findall(r'\bsi\b' , phrase_low) != []) & \
                    (re.findall(r'\bsi\s+oui\b', phrase_low) == []) & \
                    (re.findall(r'\bm[eê]me\s+si\b',phrase_low) == []))  | \
                   (re.findall( r'\b[àa]\s+condition\s+que\b' , phrase_low) != []) | \
                   (re.findall( r'\b[àa]\s+moins\s+que\b' , phrase_low) != []) | \
                   (re.findall( r'\bpour\s+peu\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\bsi\s+tant\s+est\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\bpour\s+autant\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\ben\s+admettant\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\b[àa]\s+supposer\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\ben\s+supposant\s+que\b' , phrase_low) != []) | \
                    (re.findall( r'\bau\s+cas\s+o[uù]\b' , phrase_low) != []) | \
                    (re.findall( r'\b[ée]ventuellement\b' , phrase_low) != []) | \
                    (re.findall( r'\bsuspicion\b' , phrase_low) != []) | \
                    ((re.findall( r'\bsuspect[ée]e?s?\b' , phrase_low) != []) & \
                      (re.findall( r'\bconfirm[ée]e?s?\b' , phrase_low) == [])) | \
                    (re.findall( r'\benvisag[eé]e?s?r?\b' , phrase_low) != []):

                    return 'hypothesis'
                else:
                    return 'certain'           
    
    def annotate_function(self, _input):
        
        inp = self.get_all_key_input(_input)
        
        res = []
        
        for sent in inp:
            
            if sent.attributes is None: 
                sent.attributes = {}
            sent.attributes[self.key_output] = self.detect_hypothesis(sent.value)
            
#             res.append(Annotation(
#                 type = self.key_output,
#                 value = self.detect_hypothesis(sent.value), 
#                 span = sent.span,
#                 source = self.ID,
#                 source_ID = sent.ID              
#             ))

        return res



class ATCDFamille(Annotator): 
    
    @staticmethod
    def detect_antecedent_famille(phrase):
#         patient_text = ''
#         texte_antecedent_famille = ''
        
#         phrases = re.split(r'([\r\n;\?!.])', txt)
        
#         for phrase in phrases:
        if phrase == '.':
            return None
        phrase_low = phrase.lower()

        if (len(re.findall(r'[a-z]', phrase_low) ) == 0): 
            return None

        if  (re.findall(r"\bant[eèé]c[eèé]dents\s+familiaux\b",phrase_low) != [])| \
            (re.findall(r"\bant[eèé]c[eèé]dent\s+familial\b",phrase_low) != [])| \
            (re.findall(r"\bant[eèé]c[eèé]dents\s+[a-z]+\s+familiaux\b",phrase_low) != [])| \
            (re.findall(r"\bant[eèé]c[eèé]dent\s+[a-z]+\s+familial\b",phrase_low) != [])| \
            (re.findall(r"\bhistoire\s+familiale\b",phrase_low) != [])| \
            (re.findall(r"\bm[eè]re\b",phrase_low) != [])| \
            (re.findall(r"\bp[eèé]re\b",phrase_low) != [])| \
            (re.findall(r"\bcousins?\b",phrase_low) != [])| \
            (re.findall(r"\bcousines?\b",phrase_low) != [])| \
            (re.findall(r"\btantes?\b",phrase_low) != [])| \
            (re.findall(r"\boncles?\b",phrase_low) != [])| \
            (re.findall(r"\bsoeurs?\b",phrase_low) != []) | \
            (re.findall(r"\bpapa\b",phrase_low) != []) | \
            (re.findall(r"\bmaman\b",phrase_low) != []) | \
            (re.findall(r"\bfr[eèé]res?\b",phrase_low) != []) | \
            (re.findall(r"\bgrands?[- ]parents?\b",phrase_low) != [])| \
            (re.findall(r"\bneveux?\b",phrase_low) != [])| \
            (re.findall(r"\bfils\b",phrase_low) != [])| \
            (re.findall(r"\bni[eèé]ces?\b",phrase_low) != [])| \
            (re.findall(r"\bpaternel",phrase_low) != [])| \
            (re.findall(r"\bmaternel",phrase_low) != [])| \
            (re.findall(r"\bfamille",phrase_low) != []) & (re.findall(r"avec\s+la\s+famille",phrase_low) == [])| \
            (re.findall(r"ATCDs?[^a-z]*familiaux",phrase_low) != [])| \
            (re.findall(r"ATCD?[^a-z]*familiale",phrase_low) != [])| \
            (re.findall(r"antecedents?[^a-z]*familiaux",phrase_low) != [])| \
            (re.findall(r"\bterrain *familial",phrase_low) != []) :

            return 'family'
        else:
            return 'patient'
        
        
    def annotate_function(self, _input):
        
        inp = self.get_all_key_input(_input)
        
        res = []
        
        for sent in inp:
            
            if sent.attributes is None: 
                sent.attributes = {}
            sent.attributes[self.key_output] = self.detect_antecedent_famille(sent.value)
            
#             res.append(Annotation(
#                 type = self.key_output,
#                 value = self.detect_hypothesis(sent.value), 
#                 span = sent.span,
#                 source = self.ID,
#                 source_ID = sent.ID              
#             ))

        return res

class SyntagmeTokenizer(Annotator): 
    
    @staticmethod 
    def tokenize_syntagmes(txt):
        neg_split = r'([^a-z]mais[^a-z])|(\spour\s+qu)|(\squi )|(\sentre\s)|(\scar\s+)|([\r\n\.;\?!\(\)])|(,\s+[^d][^A-Z])|( sans )|( puis )|(lequel)|(laquelle)|(hormis)|(parce\s+qu)|(bien\s+qu)|(et\s+qu)|(alors\s+qu)|(en\s+dehors\s+d)|(malgr[ée])|(en\s+raison\s+de)'
        
        phrases = re.split(neg_split,txt)
        #phrases = re.split(r'([^a-z]mais[^a-z])|(\spour\s+qu)',txt)
        phrases = [w for w in phrases if w is not None ]
        
        for i, w in enumerate(phrases):
            if (re.findall(neg_split, w) != []): 
                if phrases[i+1] != '':
                    phrases[i] += phrases[i+1]
                    phrases[i+1] = ''
                    
        phrases = [w for w in phrases if w != '' ]
        
        return phrases
    
    def annotate_function(self, _input):
        
        inp = self.get_all_key_input(_input)
        
        res = []
        
        for sent in inp:
            
            syntagmes = self.tokenize_syntagmes(sent.value)
            
            for syntagme in syntagmes:
                
                start = sent.span[0] + sent.value.find(syntagme)
                end = start + len(syntagme)-1
                
                res.append(Annotation(
                    type = self.key_output,
                    value = syntagme, 
                    span = (start, end),
                    source = self.ID,
                    source_ID = sent.ID, 
                    attributes = sent.attributes.copy()
                ))

            
#             res.append(Annotation(
#                 type = self.key_output,
#                 value = self.detect_hypothesis(sent.value), 
#                 span = sent.span,
#                 source = self.ID,
#                 source_ID = sent.ID              
#             ))

        return res    
    
    
class Negation(Annotator):

    @staticmethod
    def detect_negation(phrase):
        phrase_low = phrase.lower()
        if len(re.findall(r'[a-z]', phrase_low)) > 0:
            if (((re.findall(r"[^a-z]pas\s([a-z']*\s*){0,2}d", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z]*\s){0,2}doute", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}elimin[eé]", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}exclure", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}probl[eèé]me", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}soucis", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}objection", phrase_low) == []) & \
                  (re.findall(r"\sne reviens\s+pas", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]pas\s([a-z']*\s*){0,2}pour", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z]*\s){0,2}doute", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}pour\s+[eé]limine", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}pour\s+exclure", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]n(e\s+|'\s*)(l[ae]\s+|l'\s*)?([a-z']*\s*){0,2}pas[^a-z]", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z]*\s){0,2}doute", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}elimin[eèé]", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}exclure", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}soucis", phrase_low) == []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}objection", phrase_low) == []) & \
                  (re.findall(r"\sne reviens\s+pas", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]sans\s", phrase_low) != []) & \
                  (re.findall(r"[^a-z]sans\s+doute", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s+elimine", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s+probl[eéè]me", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s+soucis", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s+objection", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s+difficult", phrase_low) == [])) | \

                ((re.findall(r"aucun", phrase_low) != []) & \
                  (re.findall(r"aucun\s+doute", phrase_low) == []) & \
                  (re.findall(r"aucun\s+probleme", phrase_low) == []) & \
                  (re.findall(r"aucune\s+objection", phrase_low) == [])) | \

                ((re.findall(r"[^a-z][eé]limine", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}elimine", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s*([a-z']*\s*){0,2}elimine", phrase_low) == [])) | \

                ((re.findall(r"[^a-z][eé]liminant", phrase_low) != []) & \
                  (re.findall(r"[^a-z][eé]liminant\s*pas[^a-z]", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]infirm[eé]", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}infirmer", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s*([a-z']*\s*){0,2}infirmer", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]infirmant", phrase_low) != []) & \
                  (re.findall(r"[^a-z]infirmant\s*pas[^a-z]", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]exclu[e]?[s]?[^a-z]", phrase_low) != []) & \
                  (re.findall(r"[^a-z]pas\s*([a-z']*\s*){0,2}exclure", phrase_low) == []) & \
                  (re.findall(r"[^a-z]sans\s*([a-z']*\s*){0,2}exclure", phrase_low) == [])) | \

                (re.findall(r"[^a-z]jamais\s[a-z]*\s*d", phrase_low) != []) | \

                (re.findall(r"orient[eèé]\s+pas\s+vers", phrase_low) != []) | \

                (re.findall(r"orientant\s+pas\s+vers", phrase_low) != []) | \

                (re.findall(r"[^a-z]ni\s", phrase_low) != []) | \

                (re.findall(r":\s*non[^a-z]", phrase_low) != []) | \

                (re.findall(r"^\s*non[^a-z]+$", phrase_low) != []) | \

                (re.findall(r":\s*aucun", phrase_low) != []) | \

                (re.findall(r":\s*exclu", phrase_low) != []) | \

                (re.findall(r":\s*absen[ct]", phrase_low) != []) | \

                (re.findall(r"absence\s+d", phrase_low) != []) | \

                (re.findall(r"\snegati", phrase_low) != []) | \

                ((re.findall(r"[^a-z]normale?s?[^a-z]", phrase_low) != []) & \
                  (re.findall(r"pas\s+normale?s?\s", phrase_low) == [])) | \

                ((re.findall(r"[^a-z]normaux", phrase_low) != []) & \
                  (re.findall(r"pas\s+normaux", phrase_low) == [])) 
                ):
                return 'neg'
            else: 
                return 'aff'
        else: 
            return 'aff'
    
    def annotate_function(self, _input):
        
        inp = self.get_all_key_input(_input)
        
        res = []
        
        for syntagme in inp:
            
            if syntagme.attributes is None: 
                syntagme.attributes = {}
            syntagme.attributes[self.key_output] = self.detect_negation(syntagme.value)

        return []
    
    
        
class RegexMatcher(Annotator):

    def __init__(self, key_input, key_output, ID, regexp_file = "list_regexp.json"):
        
        with open(regexp_file , 'r') as f:
            self.list_regexp = json.load(f)
        super().__init__(key_input, key_output, ID)
    
    def annotate_function(self, _input): 
        
        raw = self.get_key_input(_input, 0)[0]
        syntagmes = self.get_key_input(_input, 1)
        
        if raw.value.strip() == '':
            return []
        
        all_rex = []
        
        for rex in self.list_regexp:
            
            if len(syntagmes) == 0: 
                return []
            
            # filter on document by filtre_document
            if 'filtre_document' in rex.keys() and rex['filtre_document'] != '': 
                docmatch = re.search(rex['filtre_document'], raw.value)
                if docmatch is None: 
                    continue

            for syntagme in syntagmes:
                
                all_rex += self.find_matches(rex, syntagme, raw)
            
        return all_rex

    def find_matches(self, rex, syntagme, raw, snippet_size = 60):
        res = []

        if 'casesensitive' in rex.keys() and rex['casesensitive'] == 'yes': 
            reflags = 0
        else:
            reflags = re.IGNORECASE

        for m in re.finditer(rex['regexp'], syntagme.value, flags=reflags):
            if m is not None: 

                # filter if match regexp_exclude
                if rex['regexp_exclude'] != '':
                    exclude_match = re.search(rex['regexp_exclude'], syntagme.value)
                    if exclude_match is not None: 
                        continue

                if 'index_extract' in rex.keys() and rex['index_extract'] != '': 
                    i = int(rex['index_extract'])
                else:
                    i = 0
                start = syntagme.span[0] + m.start(i)
                end = start + len(m.group(i))
                
                snippet_span = (max(start-snippet_size, 0), min(end+snippet_size, raw.span[1]))
                snippet_value = raw.value[snippet_span[0]:snippet_span[1]]
                
                res.append(Annotation(
                    type = self.key_output,
                    value = m.group(i), 
                    span = (start, end),
                    attributes = {'version':rex['version'], 'label':rex['libelle'], 'id_regexp' : rex['id_regexp'], 'snippet':snippet_value, **syntagme.attributes},
                    isEntity = True,
                    source = self.ID,
                    source_ID = syntagme.ID
                
                ))
        return res

    @staticmethod
    def add_context(matches, text, context_size=60):
        res = []
        for m in matches: 
            if m != []: 
                m = m[0]   
                start = max(m['offset_begin']-context_size, 0)
                end = min(m['offset_end']+context_size, len(text))
                m['snippet'] = text[start:end]
                res.append(m)
        return res


class QuickUMLSAnnotator(Annotator):

    def __init__(self, key_input, key_output, ID, 
                quickumls_fp = '/Users/antoine/git/QuickUMLS/umls_data/',
                overlapping_criteria = "length", # "score" or "length"
                threshold = 0.9,
                similarity_name = "jaccard", # Choose between "dice", "jaccard", "cosine", or "overlap".
                window = 5,
                accepted_semtypes = ACCEPTED_SEMTYPES): 
        
        super().__init__(key_input, key_output, ID)
        
        
        self.matcher = QuickUMLS(quickumls_fp=quickumls_fp , 
                                 overlapping_criteria=overlapping_criteria, 
                                 threshold=threshold, 
                                 window=window,
                                 similarity_name=similarity_name,
                                accepted_semtypes=accepted_semtypes)
        

    def match(self, text):
        return self.matcher.match(text)
    
    def annotate_function(self, _input):
        
        inp = self.get_all_key_input(_input)
        
        res = []
        
        for sent in inp:
            
            ents = self.match(sent.value)
            
            if sent.attributes is None: 
                sent.attributes = {}
            
            for ent in ents: 
                
                ent_attr = {'cui':ent[0]['cui'],
                           'label': ent[0]['term'],
                           'semtypes': ent[0]['semtypes'],
                           'score': ent[0]['similarity'], 
                            'snippet': sent.value
                           }
                
                start = sent.span[0] + ent[0]['start']
                end = sent.span[0] + ent[0]['end']
                
                res.append(Annotation(
                    type = self.key_output,
                    value = ent[0]['ngram'], 
                    span = (start, end),
                    source = self.ID,
                    source_ID = sent.ID, 
                    attributes = {**sent.attributes.copy() , **ent_attr}
                ))
                
        return res
          

        
class SectionSplitter(Annotator):
    def __init__(self, key_input, key_output, ID, section_dict = SECTION_DICT): 
        
        super().__init__(key_input, key_output, ID)
        
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        self.keyword_processor.add_keywords_from_dict(section_dict)
        self.head_before_treat = [ "histoire", "evolution"]
    
    def annotate_function(self, _input):
        inp= self.get_first_key_input(_input)[0]
        
        res = []
        
        match = self.keyword_processor.extract_keywords(inp.value, span_info = True)
        match = pd.DataFrame(match, columns=["match_type", "start", "end"]).sort_values(['start','end'])
        match = (match.append({"match_type": 'head', "start":0}, ignore_index=True)
                 .sort_values('start')
                 .assign(end = lambda x:x.start.shift(-1).fillna(len(inp.value)).astype('int'))
                 .assign(sl = lambda x:x.start - x.end).loc[lambda x:x.sl!=0].drop("sl", axis=1)
                 .reset_index(drop=True)
                )
        
        #set any traitement section occuring before histoire or evolution to traitement entree
        index_before_treat = match.loc[lambda x:x.match_type.isin(self.head_before_treat)].index.tolist()
        index_before_treat = min(index_before_treat, default=0)
        match.loc[lambda x:(x.match_type == "traitement")&(x.index < index_before_treat), "match_type"] = "traitement_entree"
        
        
        if inp.attributes is None:
            attributes = {}
        else:
            attributes = inp.attributes.copy()
        
        for index, row in match.iterrows():
            
            att = attributes.copy()
            att[self.key_output] = row['match_type']
            
            res.append(Annotation(
                type = self.key_output,
                value = inp.value[row['start']:row['end']],
                span = (row['start'],row['end']),
                source = self.ID,
                source_ID = inp.ID,
                attributes = att
                ))
        
        return res