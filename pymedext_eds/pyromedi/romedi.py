from rdflib import Graph, URIRef
from rdflib.term import _castLexicalToPython
from pprint import pprint
import pickle

class Romedi: 
    def __init__(self, path = None, fmt="turtle", from_cache = None):
        
        self.path = path
        self.graph = Graph()
        self.fmt = fmt
        self.from_cache = from_cache
        
        self.has_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        self.has_label = URIRef("http://www.w3.org/2000/01/rdf-schema#label")
        self.has_hidden_label = URIRef("http://www.w3.org/2004/02/skos/core#hiddenLabel")

        self.has_BNdosage = URIRef("http://www.romedi.fr/romedi/CIShasBNdosage")
        self.has_BN = URIRef("http://www.romedi.fr/romedi/BNdosagehasBN")
        self.has_PINdosage = URIRef('http://www.romedi.fr/romedi/CIShasPINdosage')
        self.has_PIN = URIRef('http://www.romedi.fr/romedi/PINdosagehasPIN')
        self.has_INdosage = URIRef('http://www.romedi.fr/romedi/PINdosagehasINdosage')
        self.has_IN = URIRef('http://www.romedi.fr/romedi/INdosagehasIN')

        self.type_CIS = URIRef("http://www.romedi.fr/romedi/CIS")
        self.type_IN = URIRef("http://www.romedi.fr/romedi/IN")
        self.type_BN = URIRef("http://www.romedi.fr/romedi/BN")

        self.has_CIP13 = URIRef("http://www.romedi.fr/romedi/CIShasCIP13")
        self.type_CIP13 = URIRef('http://www.romedi.fr/romedi/hasCIP13')

        self.has_ATC7 = URIRef("http://www.romedi.fr/romedi/CIShasATC7")
        self.type_ATC7 = URIRef('http://www.romedi.fr/romedi/hasATC7')

        self.has_ATC5 = URIRef("http://www.romedi.fr/romedi/CIShasATC5")
        self.type_ATC5 = URIRef('http://www.romedi.fr/romedi/hasATC5')

        self.has_ATC4 = URIRef("http://www.romedi.fr/romedi/CIShasATC4")
        self.type_ATC4 = URIRef('http://www.romedi.fr/romedi/hasATC4')
        
        if (self.path is not None):  
            print('parsing graph from file "{}"'.format(self.path))
            self.graph.parse(self.path, format=self.fmt)
            
            print('extracting list of CIS')
            self.cis = self.get_list(self.type_CIS)
        
        if (self.from_cache is not None): 
            self.infos = self.load_from_cache()
        else:    
            print('extracting medication infos from {} CIS'.format(len(self.cis)))
            self.infos = [self.extract_info_from_cis(x) for x in self.cis]
        
    def load_from_cache(self):
        with open(self.from_cache, 'rb') as f:
            return pickle.load(f)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.infos, f)
            

    def get_triples(self, s=None, p=None, o=None, verbose = False): 
    
        """
        if s is not None:
            s = URIRef(s)

        if o is not None:
            o = URIRef(o)

        if p is not None:
            p = URIRef(p)
        """    
        result = []
        for s_, p_, o_ in self.graph.triples((s, p, o)):   
            if verbose:
                pprint("{} - {} - {}".format(s_, p_, o_))    
            result.append((s_, p_, o_))
        return result
    
    def get_list(self, _type):
        triples = self.get_triples(s=None, 
                           p=self.has_type, 
                           o=_type)

        return [x[0] for x in triples]
    
    def get_info(self, s,p):
        infos = []
        for _,_,o in self.get_triples(s=s, p=p):
            infos.append(o)
        return(infos)

    def get_info_from_list(self,list_s, p):
        info_list = []
        for s in list_s:
            info_list += self.get_info(s,p)
        return(info_list)
    
    
    def extract_info_from_cis(self, sel_cis):

        res = {'cis': sel_cis.toPython()}
        CIS_label = self.get_info(sel_cis, self.has_label)
        res['CIS_label'] = [x.toPython() for x in CIS_label]

        BNdosage = self.get_info(sel_cis, self.has_BNdosage)
        #pprint(get_triples(BNdosage[0]))
        BNdosage_label = self.get_info_from_list(BNdosage, self.has_label)
        res['BNdosage_label'] = [x.toPython() for x in BNdosage_label]


        BN = self.get_info_from_list(BNdosage, self.has_BN)
        #pprint(get_triples(BN[0]))
        BN_label = self.get_info_from_list(BN, self.has_label)
        res['BN_label'] = [x.toPython() for x in BN_label]

        BN_hidden_label = self.get_info_from_list(BN, self.has_hidden_label)
        res['BN_hidden_label'] = [x.toPython() for x in BN_hidden_label]

        PINdosage = self.get_info(sel_cis, self.has_PINdosage)
        #pprint(get_triples(PINdosage[0]))
        PINdosage_label = self.get_info_from_list(PINdosage, self.has_label)
        res['PINdosage_label'] = [x.toPython() for x in PINdosage_label]

        PIN = self.get_info_from_list(PINdosage, self.has_PIN)
        PIN_label = self.get_info_from_list(PIN, self.has_label)
        res['PIN_label']= [x.toPython() for x in PIN_label ]

        INdosage = self.get_info_from_list(PINdosage ,self.has_INdosage)
        INdosage_label = self.get_info_from_list(INdosage, self.has_label)
        res['INdosage_label'] = [x.toPython() for x in INdosage_label]

        IN = self.get_info_from_list(INdosage, self.has_IN)
        IN_label = self.get_info_from_list(IN, self.has_label)
        res['IN_label'] = [x.toPython() for x in IN_label]

        IN_hidden_label = self.get_info_from_list(IN, self.has_hidden_label)
        res['IN_hidden_label'] = [x.toPython() for x in IN_hidden_label]

        CIP13_list = self.get_info(sel_cis, self.has_CIP13)
        if (len(CIP13_list) > 0):
            CIP13 = self.get_info_from_list(CIP13_list, self.type_CIP13)
            res['CIP13'] = CIP13[0].toPython()
        else:
            res['CIP13'] = None

        #pprint(res)
        res['ATC7'] = None
        ATC7_list = self.get_info(sel_cis, self.has_ATC7)
        #print(ATC7_list)
        if len(ATC7_list) > 0:
            ATC7 = self.get_info_from_list(ATC7_list, self.has_label)
            if len(ATC7) > 0:
                res['ATC7'] = ATC7[0].toPython()


        res['ATC5'] = None
        ATC5_list = self.get_info(sel_cis, self.has_ATC5)
        if len(ATC5_list) > 0:
            ATC5 = self.get_info_from_list(ATC5_list, self.has_label)
            if len(ATC5)> 0:
                res['ATC5'] = ATC5[0].toPython()

        res['ATC4'] = None        
        ATC4_list = self.get_info(sel_cis, self.has_ATC4)
        if len(ATC4_list) > 0:
            ATC4 = self.get_info_from_list(ATC4_list, self.has_label)
            if len(ATC4)> 0:
                res['ATC4'] = ATC4[0].toPython()
        else:
            res['ATC4'] = None

        return(res)
    
   
