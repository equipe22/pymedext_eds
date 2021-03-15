import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import itertools
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
import numpy as np
from .normalisation import clean_freq, clean_dose, clean_drug, clean_class


##########################################################################################################################
################################################ GENERAL USEFULL FONCTIONS FOR METRICS ###################################
##########################################################################################################################

def get_neg_score(conll):
    
    neg_scores = [float(score) if pred !="O" else 1-float(score)  for score, pred in zip(conll["neg_pred_score"], conll["neg_pred"])]

    return sum(neg_scores) / len(neg_scores)


def conll_2_brat(pdf, col_lab):
    """
    given a conll dataframe return the brat format of match tags
    """
    inside_ind = pdf.loc[:, col_lab] !="O"
    pdf = pdf.loc[inside_ind]

    pdf = (pdf.assign(tag_id = lambda x:(x.loc[:, col_lab].str[0] == "B").cumsum())
           .assign(tag_type = lambda x:x.loc[:,col_lab].str[2:])
          )

    brat = {col_lab +"__"+ str(k[0]):{"start":group.start.min(), 
                                      "end":group.end.max(),
                                      "tag":k[1], 
                                      "tok":group.tok.tolist(), 
                                      "probas":group.loc[:,col_lab+"_score"].tolist() if col_lab+"_score" in group.columns else [],
                                      "neg_score":get_neg_score(group) if "neg_pred_score" in group.columns else 0,
                                      "is_neg":group.loc[:,"neg_tag"].iloc[0][2:]=="NEG" if "neg_tag" in group.columns else False,


                                     } \
            for k, group in pdf.groupby(["tag_id", "tag_type"])}


    
    return brat


def custom_agg(series, digits = 1):
    series = series * 100
    mu = np.mean(series).round(digits)
    mu = str(mu)
    low, high = np.nanquantile(series, [0.05, 0.95])
    low = str(round(low, digits))
    high = str(round(high, digits))

    ci = mu + " [" + low + "-" + high + "]"

    return ci

##########################################################################################################################
############################################ RELATION FONCTIONS ##########################################################
##########################################################################################################################

def check_interval(child_span, parent_span):
    """
    Given a child span and a parent span, check if the child is inside the parent
    """
    child_start, child_end = child_span
    parent_start, parent_end = parent_span
    if (
        (child_start >= parent_start)
       &(child_end <= parent_end)
       ):
        return True
    else:
        return False
    
    
def parent_relation(child_span_0, child_span_1, parent_span):
    """
    Given two child span, and one parent span, check if both children are included in the parent
    """
    if check_interval(child_span_0, parent_span) & check_interval(child_span_1, parent_span):
        return True
    else:
        return False

def get_tags_from_match(child_span_0, child_span_1, tags):
    """
    Given two entities spans,
    check if both are within one of the tags span,
    and return the first match or O
    """
    match_tags = []
    for k, v in tags.items():
        parent_span = (v["start"], v["end"])
        if parent_relation(child_span_0, child_span_1, parent_span):
            match_tags.append(v["tag"])
            
    return match_tags[0] if match_tags else "O"

def get_candidate_entities(entities_list):
    """
    Given a list of entities dict, generate the candidate entities pairs
    Args:
        entities_list: list of entities dict
    """
    #merge entities & events
    candidate = {k: v for d in entities_list for k, v in d.items()}
    
    #candidate are generated as the combination of all possible pairs of entities/events without order
    candidate = [(t[0], t[1],
                  candidate[t[0]]['start'], candidate[t[0]]['end'],
                  candidate[t[1]]['start'], candidate[t[1]]['end'],
                  candidate[t[0]]['tag'], candidate[t[1]]['tag']
                 ) \
                 for t in itertools.combinations(candidate.keys(), 2)]

    candidate = pd.DataFrame(candidate, columns=['tag_0', 'tag_1', 'start_0', 'end_0', 'start_1', 'end_1', 'type_0', 'type_1'])
    
    return candidate
    


def get_relation_tag(candidate, relation_spans):
    """
    Given candidate entities pairs, and a list of relations, tag the entities pairs with these relations
    """
    tags = []
    for i, row in candidate.iterrows():
        #tag entity pair as related if included in relation span
        child_0 = (row.start_0, row.end_0)
        child_1 = (row.start_1, row.end_1)
        tag = get_tags_from_match(child_0, child_1, relation_spans)
        tags.append(tag)
        
    return tags


def brat_to_relations(group, entity_gold, entity_pred, relation_gold, relation_pred):
    """
    Given a sentence, returns the combination of candidate entities pairs and their relations
    """
    #from sentence in conll to labels in brat
    #entities/events
    entity_bag = [conll_2_brat(group, col) for col in entity_gold]
    entity_bag_pred = [conll_2_brat(group, col) for col in entity_pred]

    #relations
    drugblob_pred = conll_2_brat(group, relation_pred)
    drugblob_tag = conll_2_brat(group, relation_gold)

    #generate dataframe of candidate entities pairs,
    #from true entities and predicted entities
    candidate = get_candidate_entities(entity_bag)
    candidate_pred = get_candidate_entities(entity_bag_pred)

    #tag entities pair with relation label
    gold_tag = get_relation_tag(candidate, drugblob_tag)
    oracle_tag = get_relation_tag(candidate, drugblob_pred)
    pred_tag = get_relation_tag(candidate_pred, drugblob_pred)

    #reformat
    candidate = candidate.assign(gold_tag = gold_tag).assign(oracle_tag = oracle_tag).drop(['tag_0', 'tag_1'], axis=1)
    candidate_pred = candidate_pred.assign(pred_tag = pred_tag).drop(['tag_0', 'tag_1'], axis=1)

    #define relations index by span and type,
    #merge to assign to each unique pairs a gold relation label, oracle and pred labels²
    candidate_pred = candidate_pred.set_index(['start_0','end_0', 'start_1', 'end_1', 'type_0', "type_1"])
    candidate = candidate.set_index(['start_0','end_0', 'start_1', 'end_1', 'type_0', "type_1"])
    candidate = candidate.join(candidate_pred, how="outer").fillna("O").reset_index()
    
    #assign sentence_id and section_tag
    if not candidate.empty:
        candidate = (candidate
                     .assign(sentence_id = group.iloc[0].sentence_id)
                     .assign(section_type = group.iloc[0].section_tag[2:])
                    )
    
    return candidate


def get_relations(data,  entity_gold, entity_pred, relation_gold, relation_pred, n_job=20):
    """
    Get from a conll dataframe, within each sentence, the overall pair of entities that true or predicted, and the labels of their relations
    Args:
        data (pd.Dataframe):
            One token per row
            a start column, for start span
            a end column, for end span
            
        entity_gold (list[str]): a list of columns names for gold entities
        entity_pred (list[str]): a list of columns names for predicted entities
        relation_gold (str): column name for gold relation
        relation_pred (str): column name for pred relation

    Returns:
        relations_acc (pd.Dataframe):  a relations dataframe contains one relations between two entities per row
                            
    """
    
    if n_job == 1:
        relations_acc = []
        for k, sentence in data.groupby('sentence_id'):
            relations_acc.append(brat_to_relations(sentence,  entity_gold, entity_pred, relation_gold, relation_pred))

    else:
        relations_acc = Parallel(n_jobs=n_job)(delayed(brat_to_relations)\
                                 (sentence,  entity_gold, entity_pred, relation_gold, relation_pred)\
                                        for k, sentence in data.groupby('sentence_id'))
    

    relations_acc = [t for t in relations_acc if not t.empty]
    relations_acc = pd.concat(relations_acc)
    assert relations_acc.duplicated(subset=['start_0', 'end_0', 'start_1', 'end_1', 'type_0', 'type_1', 'sentence_id']).sum() == 0
    
    return relations_acc


def get_metrics(relations, gold_tag, pred_tag, filter_tn=True, any_filter = ['ENT/DRUG', 'ENT/CLASS'], section_filter = []):
    """
    Compute metrics from relations dataframe
    Args:
        relations (pd.Dataframe): a relations dataframe contains one relations between two entities per row,
                                  with at leat two columns for gold label and predicted label
                                  optional: a column for section_id if you want to filter in a list of section
                                  optionnal: two column, type_0 and type_1 for each type of entity
                                  
        gold_tag (str): column name of gold label
        pred_tag (str): column name of predicted label
        filter_tn (bool): wether to filter out true negatives relations
        any_filter (list[str]): list of entities that must be present as one of the two type of entities
        section_filter (list[str]): list of sections type to filter
    
    Returns:
        metrics (pd.Dataframe): a single row dataframe with metrics
                                  
    """
    
    #apply filters    
    if any_filter:
        #relation rule 0: there cannot be a drug/drug or class/drug relation
        relations = relations.loc[lambda x:~x.loc[:,["type_0", "type_1"]].isin(any_filter).all(1)]
        #relation rule 1: relation is defined between a drug/class and another field/event
        relations = relations.loc[lambda x:x.loc[:,["type_0", "type_1"]].isin(any_filter).any(1)]
    if section_filter:
        #filter to get results on a specific section type
        relations = relations.loc[lambda x:x.loc[:,"section_type"].isin(section_filter)]
        sections_tag = "/".join(section_filter)
    else:
        sections_tag = "All"
        
    #drop true neg
    if filter_tn:
        relations = relations.loc[lambda x:~((x.loc[:,pred_tag]=="O")&(x.loc[:,gold_tag]=="O"))]

    #get true labels
    labels = pd.unique(relations.loc[:,["gold_tag","pred_tag"]].values.ravel('K'))
    labels = list(labels)
    assert len(labels) == 2, "does not yet take mutli-label relations into account"
    pos_label = [t for t in labels if t!="O"][0]

    #compute metrics
    gold = relations.loc[:,gold_tag]
    pred = relations.loc[:,pred_tag]
    tp, fn, fp, tn = confusion_matrix(gold, pred, labels = [pos_label, "O"]).ravel()
    acc = accuracy_score(gold, pred)
    f1= f1_score(gold, pred, average="binary", pos_label=pos_label)
    r = recall_score(gold, pred, average="binary", pos_label=pos_label)
    p = precision_score(gold, pred, average="binary", pos_label=pos_label)
    
    #reformat metrics
    metrics = pd.DataFrame([[tp + fn, tp, fn, fp, r, p, f1, acc]],
                           columns=['total_pos', "TP", "FN", "FP", "Recall", "Precision", "F1", "Accuracy"],
                           index=[pos_label])
    
    metrics = metrics.assign(section_type = sections_tag).assign(pred_type = pred_tag)

    return metrics


def bootstrappred_metrics(relations,
                          gold_tag,
                          pred_tag,
                          any_filter = ['ENT/DRUG', 'ENT/CLASS'],
                          section_filter=[],
                          n_bootstrap = 1000):
    np.random.seed(42)
    seeds = np.random.randint(100000000, size= n_bootstrap)
    sample_metrics = []
    for seed in seeds:
        sample = relations.sample(frac=1, replace=True, random_state=seed)
        tmp = get_metrics(sample, 
                gold_tag = gold_tag,
                pred_tag = pred_tag,
                any_filter = any_filter,
                section_filter=section_filter)

        sample_metrics.append(tmp)


    sample_metrics = pd.concat(sample_metrics)
    sample_metrics = sample_metrics.loc[:, ["Recall", "Precision", "F1", "Accuracy"]].agg(custom_agg)
    sample_metrics.index = sample_metrics.index +"_ci"

    overall_metrics = get_metrics(relations, 
                                  gold_tag = gold_tag,
                                  pred_tag = pred_tag,
                                  any_filter = any_filter,
                                  section_filter=section_filter)


    overall_metrics = pd.concat([overall_metrics, sample_metrics.to_frame(name=overall_metrics.index[0]).T], axis=1, sort = False)


    return overall_metrics



##########################################################################################################################
################################################## ENTITIES FUNCTIONS ####################################################
##########################################################################################################################

def equal_ent(child_0, child_1):
    """
    Check if two span are equal
    """
    if (child_0[0] == child_1[0])&(child_0[1] == child_1[1])&(child_0[2] == child_1[2]):
        return True
    else:
        return False

def ent_isin(child, candidates):
    """
    Check if a span is in a list of candidate span
    """
    for cand in candidates:
        if equal_ent(child, cand):
            return True
    else:
        return False




def get_entity_counts(data, gold_col, pred_col):
    """
    get entities for each sentence
    """
    correct_acc, incorrect_acc = [], []
    for k, sentence in data.groupby('sentence_id'):
        correct_pred, incorrect_pred  = compare(sentence, gold_col, pred_col)
        correct_acc.append(correct_pred)
        incorrect_acc.append(incorrect_pred)

    correct_acc = pd.concat(correct_acc, sort=False)
    incorrect_acc = pd.concat(incorrect_acc, sort=False)
    
    return correct_acc, incorrect_acc



def get_entities(sentence, gold_col, pred_col, only_has_norm = False, class_norm = None, romedi = None):
    """
    For a given sentence; get predicted and gold entities with one row per entities
    """
    #get entitities list
    pred = conll_2_brat(sentence, pred_col)
    tag = conll_2_brat(sentence, gold_col)

    #reformat
    if only_has_norm:
        pred = [(t['start'], t['end'], t['tag']) for t in pred.values() if has_norm(" ".join(t['tok']), t["tag"], class_norm = class_norm, romedi = romedi)]
    else:
        pred = [(t['start'], t['end'], t['tag']) for t in pred.values()]

        
    tag = [(t['start'], t['end'], t['tag']) for t in tag.values()]


    tag = pd.DataFrame(tag, columns=["start", "end", "gold"]).set_index(["start", "end"])
    pred = pd.DataFrame(pred, columns=["start", "end", "pred"]).set_index(["start", "end"])

    entities = tag.join(pred, how="outer").fillna("O").reset_index()

    entities = (entities
                .assign(sentence_id = sentence.iloc[0].sentence_id)
                .assign(section_type = sentence.iloc[0].section_tag[2:])
               )
    
    return entities


def compute_entities_metrics(entities, pred_tag, section_filter=[]):
    """
    Compute metrics given list of correct and incorect entities
    """
            
    if section_filter:
        entities = entities.loc[lambda x:x.loc[:,"section_type"].isin(section_filter)]
        sections_tag = "/".join(section_filter)
    else:
        sections_tag = "All"
        
#     assert entities.loc[lambda x:(x.gold=='O')&(x.pred=='O')].empty

    gold = entities.gold
    pred = entities.pred

    labels = gold.unique().tolist()

    #get confusion matrics
    cm = confusion_matrix(gold, pred, labels=labels)
    cm = pd.DataFrame(cm, columns=labels, index=labels)

    #compute type wise metrics
    tp = np.diag(cm)
    total_pos = cm.sum(1)
    total_pred = cm.sum(0)
    fn = total_pos - tp
    fp = total_pred - tp
    recall = tp/(tp+fn + 1e-8)
    precision = tp/(tp+fp + 1e-8)
    f1 = (2*(recall*precision)/(recall+precision + 1e-8))
    #reformat
    metrics = pd.concat([total_pos, total_pred, fn, fp, recall, precision, f1], axis=1)
    metrics.columns = ["total_pos", "total_pred", "fn", "fp", "Recall","Precision", "F1"]
    metrics = metrics.assign(tp = tp)
    if 'O' in metrics.index:
        metrics = metrics.drop('O')

    #compute overall metrics
    overall = metrics.sum(0)
    overall_recall =  overall.tp / overall.total_pos
    overall_precision =  overall.tp / overall.total_pred
    
    micro_average_lab = metrics.index[0].split('/')[0] + "/micro-average"

    metrics.loc[micro_average_lab] = [int(overall.total_pos),
                                int(overall.total_pred),
                                int(overall.fn),
                                int(overall.fp),
                                overall_recall,
                                overall_precision,
                                (2*(overall_recall*overall_precision)/(overall_precision+overall_recall)),
                                int(overall.tp)
                               ]
    
    #reformat
    metrics = metrics.assign(section_type = sections_tag).assign(pred_type = pred_tag)
    
    return metrics

def has_norm(mention, ent_type, context= "", class_norm = None, romedi = None):
    if ent_type == "ENT/FREQ":
        cleaned =  clean_freq(mention)
        return cleaned != 'NA'

    elif ent_type == "ENT/DOSE":
        cleaned =  clean_dose(mention)
        return cleaned != ''

    elif ent_type == "ENT/DRUG":
        cleaned =  clean_drug(mention, romedi)
        return cleaned != {}
    elif ent_type == 'ENT/CLASS':
        cleaned = clean_class(mention, class_norm)
        return cleaned !=[]

    else:
        return False
    
def get_all_entities(data, gold_col, pred_col, n_job=20, only_has_norm = False, class_norm = None, romedi = None):
    
    if n_job == 1:
        all_entities = []
        for k, sentence in data.groupby('sentence_id'):
            all_entities.append(get_entities(sentence, gold_col, pred_col, only_has_norm, class_norm = class_norm, romedi = romedi))
            
    else:
        all_entities = Parallel(n_jobs=n_job)(delayed(get_entities)\
                                     (sentence, gold_col, pred_col, only_has_norm, class_norm = class_norm, romedi = romedi)\
                                            for k, sentence in data.groupby('sentence_id'))

    all_entities = [t for t in all_entities if not t.empty]

    all_entities = pd.concat(all_entities)
    
    return all_entities


def bootstrappred_entities_metrics(all_entities,
                                   pred_tag,
                                   section_filter=[],
                                   n_bootstrap = 1000):
    np.random.seed(42)
    seeds = np.random.randint(100000000, size= n_bootstrap)
    sample_metrics = []
    for seed in seeds:
        sample_entities = all_entities.sample(frac=1, replace=True, random_state=seed)
        tmp = compute_entities_metrics(sample_entities , pred_tag, section_filter)
        sample_metrics.append(tmp)


    sample_metrics = pd.concat(sample_metrics)
    sample_metrics = sample_metrics.dropna()
    overall_metrics =  compute_entities_metrics(all_entities, pred_tag, section_filter)


    sample_metrics = sample_metrics.loc[:, ["Recall", "Precision", "F1"]].groupby(sample_metrics.index).agg(custom_agg)

    sample_metrics.columns = sample_metrics.columns + "_ci"

    overall_metrics = pd.concat([overall_metrics, sample_metrics], sort=False, axis=1)
    return overall_metrics