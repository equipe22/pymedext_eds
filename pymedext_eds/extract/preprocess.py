import pickle
from flashtext import KeywordProcessor
import pandas as pd
import re
import sentencepiece as spm
from unidecode import unidecode
import numpy as np
from nltk.tokenize import RegexpTokenizer
import unicodedata
from flair.embeddings import BertTokenizer
import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError
    


class SectionSplitter(object):
    def __init__(self, 
                 terminology_path:str="",
                 mode="section_v1"):
        
        with open(terminology_path, "rb") as f:
            section_dict = pickle.load( f)
            
        self.path = terminology_path
        
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        self.keyword_processor.add_keywords_from_dict(section_dict)
        self.head_before_treat = [ "histoire", "evolution"]
        self.mode = mode

            
    def transform_text(self, text):
        match = self.keyword_processor.extract_keywords(text, span_info = True)
        match = pd.DataFrame(match, columns=["match_type", "start", "end"]).sort_values(['start','end'])
        match = (match.append({"match_type": 'head', "start":0}, ignore_index=True)
                 .sort_values('start')
                 .assign(end = lambda x:x.start.shift(-1).fillna(len(text)).astype('int'))
                 .assign(sl = lambda x:x.start - x.end).loc[lambda x:x.sl!=0].drop("sl", axis=1)
                 .reset_index(drop=True)
                )
        
        if self.mode == "section_v2":
            #set any traitement section occuring before histoire or evolution to traitement entree
            index_before_treat = match.loc[lambda x:x.match_type.isin(self.head_before_treat)].index.tolist()
            index_before_treat = min(index_before_treat, default=0)
            match.loc[lambda x:(x.match_type == "traitement")&(x.index < index_before_treat), "match_type"] = "traitement_entree"

        return match
    
    
class TerminologyFeaturizer(object):
    def __init__(self, terminology_path:str="", key_subset=[], lower_text=True):
        with open(terminology_path, "rb") as f:
            section_dict = pickle.load( f)
        
        if key_subset:
            section_dict = {k:v for k,v in section_dict.items() if k in key_subset}
          
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.keyword_processor.add_keywords_from_dict(section_dict)
        self.lower_text =  lower_text
        

            
    def transform_text(self, text):
        text = text.lower() if self.lower_text else text
        match = self.keyword_processor.extract_keywords(text, span_info = True)
        match = pd.DataFrame(match, columns=["match_type", "start", "end"]).sort_values(['start','end'])

        return match

class SentenceSplitter(object):
    def __init__(self, mode = "pattern"):
        if mode == "pattern":
            self.pattern = re.compile('[\n|\.|\?]\s?(?=\-?\s?[A-Z])')
        elif mode == "pattern_v2":
            self.pattern = re.compile('[\n|\.|\?]\s+(?=\-?\s?[A-Z])')

            
    def transform_text(self, text):
        match = [m.span() for m in re.finditer(self.pattern, text)]
        match = [0] + [t[1] for t in match] + [len(text)]
        match = [{"start":match[i], "end":match[i+1]} for i in range(len(match)-1)]
        match = pd.DataFrame(match, columns=["start", "end"]).assign(match_type = "sentence").sort_values(['start', 'end'])

        
        return match
    
    





class Tokenizer(object):
    def __init__(self, mode="pattern", model_path = "", perform_checks= False):
        self.mode = mode
        self.perform_checks = perform_checks
        if mode == "pattern":
            self.pattern = re.compile("[dnl]['´`]|\w+|\$[\d\.]+|[^\r\n\t\f\v\/\)\(\[\]\)\- ]+")
            self.tokenizer = RegexpTokenizer(self.pattern)
        elif mode == "pattern_v2":
            tokens_patt = [
                        "[A-Z](?:\.[A-Z])+", #sigles  A.B.C
                        "[nN]°", #N°
                        "\d+(?:[,\.\\/:]\d+)+", #date 20.20.2020 or dose 20.5 mg
                        "[Qq]u['´`’]", #Qu'elle
                        "[dnmljDNMLJ]['´`’]", #d'une 
                        "\w+", #
                        "\$[\d\.]+", #
                        "[^\r\n\t\f\v\/\)\(\[\]\)\-,+: ]+" # word separators
                                  ]

            self.pattern = re.compile("|".join(tokens_patt))
            self.tokenizer = RegexpTokenizer(self.pattern)
            
        elif mode == "sp":
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(model_path)
        else:
            raise NotImplementedError
            
            
    def transform_text(self, text):
        if self.mode in ["pattern", "pattern_v2"]:
            tokens = pd.DataFrame([(text[t[0]:t[1]], t[0], t[1]) for t in self.tokenizer.span_tokenize(text)], columns=['tok', 'start', 'end'])

        elif self.mode == "sp":
            tokens = self.tokenizer.EncodeAsPieces(text)
            tokens = (pd.DataFrame(tokens, columns=['tok'])
                      .assign(has_underscore = lambda x:(x.tok.str[0]=="▁").astype('int'))
                      #legnth of tokens as str len minux has underscore
                      .assign(length = lambda x:x.tok.str.len() - x.has_underscore) 
                      #starts as cum sum of tokens length and space, don't support multi space
                      .assign(start= lambda x:((x.length.shift().fillna(-1).astype('int') + x.has_underscore).cumsum())) 
                      .assign(end= lambda x:x.start + x.length)
                      .drop(["length", "has_underscore"], axis=1)
                     )
        else:
            raise NotImplementedError
            
        if self.perform_checks:
            self.check(text, tokens)

        tokens.tok = tokens.tok.astype('str')
        return tokens
    
    def decode(self, tokens_list):
        if self.mode == "sp":
            string = self.tokenizer.DecodePieces(tokens_list)
        elif self.mode == "pattern":
            string = " ".join(tokens_list)
        else:
            raise NotImplementedError
        return string
        
    
    def check(self, text, tokens):
        for i, row in tokens.iterrows():
            compare_string(row.tok, text[row.start:row.end])

            
class TextPreprocessing(object):
    def __init__(self, 
                 section_kwargs, 
                 sentence_kwargs, 
                 tokenization_kwargs, 
                 terminology_featurizer_kwargs = {}, 
                 section_wise = True,
                 sentence_wise = True, 
                 text_cleaner = lambda x:x, 
                 perform_checks = False, 
                 concat_small = False, #deprecated
                 bert_tokenizer = None,
                 deprecated=True, #temporary
                 max_sent_len = 300):
        
        self.section_splitter = SectionSplitter(**section_kwargs)
        self.sentence_splitter = SentenceSplitter(**sentence_kwargs)
        self.tokenizer = Tokenizer(**tokenization_kwargs)
        
        if terminology_featurizer_kwargs:
            self.terminology_featurizer = TerminologyFeaturizer(**terminology_featurizer_kwargs)
        else:
            self.terminology_featurizer = None
        self.section_wise = section_wise
        self.sentence_wise = section_wise
        self.perform_checks = perform_checks
        self.max_sent_len = max_sent_len #maximum sentence length in tokens

        if isinstance(text_cleaner, str):
            if text_cleaner == "bert_clean_text":
                self.text_cleaner = bert_clean_text
            elif text_cleaner == "bert_clean_text2":

                self.text_cleaner = bert_clean_text2
            else:
                raise NotImplementedError
        else:
            self.text_cleaner = text_cleaner
        self.use_bert = False
        if bert_tokenizer:
            self.use_bert = True
            self.bert_clean_text = bert_clean_text

            self.bert_tokenizer = BertTokenizer(bert_tokenizer)
        else:
            self.bert_clean_text = False

        self.deprecated = deprecated
        
    def transform(self, text_list):
        match_list = []
        for text in text_list:
            match_list.append(self.transform_text(text))
        return match_list
    
    def transform_text(self, text, tags_dict = {}):
        with timeout(60):
            return self._transform_text(text, tags_dict)
            
    def _transform_text(self, text, tags_dict= {}):

        #0. text default
        text = text.strip()
        if self.bert_clean_text and self.deprecated:
            text = bert_clean_text(text)

        #1. get sections spans
        section_match = self.section_splitter.transform_text(text)
        
        #2. get sentence spans      #TODO compositionalize this      
        if self.section_wise:
            all_sentences_match = []
            for i, row in section_match.iterrows():
                section_text = text[row.start:row.end]
                sentence_match = self.sentence_splitter.transform_text(section_text)
                sentence_match = (sentence_match
                                  .assign(start = lambda x:x.start + row.start)
                                  .assign(end = lambda x:x.end + row.start)
                                  .assign(section_id = i)
                                 )    
                all_sentences_match.append(sentence_match)
            all_sentences_match = pd.concat(all_sentences_match, sort=False).reset_index(drop=True)
            
        else:
            all_sentences_match = self.sentence_splitter.transform_text(text)
        
        #3. get tokens spans
        if self.sentence_wise:
            all_tokens = []
            for i, row in all_sentences_match.iterrows():
                sentence_text = text[row.start:row.end]
                tokens = self.tokenizer.transform_text(sentence_text)
                tokens = (tokens
                          .assign(start = lambda x:x.start + row.start)
                          .assign(end = lambda x:x.end + row.start)
                          .assign(sentence_id = i)
                         )    
                all_tokens.append(tokens)
            all_tokens = pd.concat(all_tokens).reset_index(drop=True)
            
        else:
            all_tokens = self.tokenizer.transform_text(text)
            
        #4. Get terminology/RBS features
        if self.terminology_featurizer is not None:
            terminologies_match = self.terminology_featurizer.transform_text(text)
        else:
            terminologies_match = {}
            
        #4. Align labels with tokens
        tags_dict = {**{"section_tag":section_match,
                        "sentence_tag":all_sentences_match,
                        "terminology_tag":terminologies_match},
                     **tags_dict}
        
        for key, tags in tags_dict.items():
            all_tokens = self.span_align(key, all_tokens, tags)
            
        #5. clean version of text
        all_tokens = all_tokens.assign(clean_text = lambda x:x.tok.apply(self.text_cleaner))
        

        #6.post process: max seq len
        all_tokens = self.cut_max_tag_len(all_tokens)
        
        if self.use_bert and self.deprecated:
#             assert len(re.findall("\n", text)) < 500, "too much carriage return"
            assert len(self.bert_tokenizer.tokenize(" ".join(all_tokens['tok']))) < 10 * len(all_tokens), "expect issue with tokenizations"
            all_tokens = self.recursive_split(all_tokens)
            
        if self.perform_checks:
            assert pd.Series(all_tokens[lambda x:x.sentence_tag.str[0]=="B"].index).diff().max() < self.max_sent_len, "Sentence length must be lower than {}".format(self.max_sent_len)
            


        return all_tokens, text

    def span_align(self, col_name, tokens, tags, default_tag = "O"):
        tags = tags.reset_index(drop=True)
        insert_start = np.searchsorted(tokens.start, tags.start, side='left')
        insert_stop = np.searchsorted(tokens.end, tags.end, side= 'left')

        tokens[col_name] = default_tag
        for i, (start, end, match_type) in enumerate(zip(insert_start, insert_stop, tags.match_type)):
            tokens.loc[start:end, col_name] = ["B-"+match_type] + ["I-"+match_type]*(len(tokens.loc[start:end, col_name]) -1)
            if self.perform_checks:
                candidate = self.tokenizer.decode(tokens.loc[start:end, "tok"].tolist())
                if "mention" in tags.columns:
                    mention = tags.loc[i, "mention"]
                    compare_string(candidate, mention)

        return tokens
    
    def cut_max_tag_len(self, df, tag= "sentence_tag"):
        #if using bert, compute number of tokens given bert tokenization
        if self.use_bert:
            df = df.assign(n_tok = lambda x:x.clean_text.apply(self.bert_tokenizer.tokenize).apply(len))
            
        #else 1 token is 1 token
        else:
            df["n_tok"] = 1
            
        tmp_len = 0
        for i, row in df[tag].iteritems():
            tmp_len += df.loc[i, "n_tok"]
            #sentence len begin at 1
            if row[0]=="B":
                #reset count
                tmp_len = df.loc[i, "n_tok"]
            #cut sentence if it reach max sent len
            if tmp_len >= self.max_sent_len:
                #add sentence split
                df.loc[i, tag] = "B" + df.loc[i, tag][1:]
                #reset count
                tmp_len = df.loc[i, "n_tok"]

                
        df = df.drop("n_tok", axis=1)
        #recompute sentence ids
        df = df.assign(sentence_id = (df.sentence_tag.str[0] == "B").cumsum())

        return df

    def recursive_split(self, conll):
        for sent_id, group in conll.groupby("sentence_id"):
            bert_len = len(self.bert_tokenizer.tokenize(" ".join(group['tok'])))
            #if too long sentence
            if bert_len > self.max_sent_len:
                #split sentence in num_split
                num_split = (bert_len // self.max_sent_len)  + 1
                new_sent = []
                for sent in np.array_split(group['sentence_tag'], num_split):
                    sent.iloc[0] = 'B-sentence'
                    new_sent.append(sent)
                group.loc[group.index, 'sentence_tag'] = pd.concat(new_sent).tolist()
                conll.loc[group.index, 'sentence_tag'] =  group.sentence_tag


                #redefine sentence ids
                conll = conll.assign(sentence_id = (conll.sentence_tag.str[0] == "B").cumsum())
                self.recursive_split(conll)

        return conll    


        
def compare_string(candidate, gold):
    candidate = clean_string(candidate)
    gold = clean_string(gold)
    assert candidate == gold, print("'{}' is different from '{}'".format(candidate , gold))
    
    
def clean_string(string):
    string = string.replace("▁", '')
    string = re.sub("[^A-Za-z]", " ", string).strip().lower()
    string = re.sub(' +', ' ', string)
    string = unidecode(string)

    return string


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def bert_clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    text = re.sub('[_—.]{4,}', '__', text)

    text = unicodedata.normalize("NFKC", text)
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
#         elif _is_whitespace(char):
#             output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def bert_clean_text2(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    text = re.sub('[_—.]{4,}', '__', text)

    text = unicodedata.normalize("NFKC", text)
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        elif _is_whitespace(char):
            output.append("_")
        else:
            output.append(char)
    text = "".join(output)
    return text if text else "_"