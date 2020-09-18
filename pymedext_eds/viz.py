from spacy import displacy

def get_dict_annotations(document, _type, source_id = None, target_id = None):
        res = []
        for anno in document['annotations']:
            if source_id is not None: 
                if anno['source_ID'] != source_id:
                    continue
            if target_id is not None:
                if anno['target_ID'] != target_id:
                    continue
            if anno['type'] == _type:
                res.append(anno)
        return res

def convert_to_displacy(document, entity_type, source_id, attributes):
    ents= []
    annots = get_dict_annotations(document, entity_type, source_id = source_id)
    #annots = [x for x in document if x['type'] in entity_type and x['source_ID'] == source_id]
    if len(annots) > 0:
        for annot in annots:
            ents.append({"start": annot['span'][0], 'end':annot['span'][1], "label": entity_type.upper()})
            drug_id = annot['id']
            if attributes != []:
                for att, val in annot['attributes'].items(): 
                    if att in attributes:
                        for v in val:
                            ents.append({"start":v['span'][0],'end':v['span'][1], 'label':att.upper()})
    return ents


def display_annotations(document, root = "sentence", entities = ["ENT/DRUG", "ENT/DOSE"], attributes = ['ENT/DOSE'],
                        palette = [ '#ffb3ba','#ffdfba','#ffffba','#baffc9','#bae1ff']):

    to_disp = []
    tmp = { "ents": []}

    for sent in get_dict_annotations(document,root):

        for entity in entities:
            conv_ents = convert_to_displacy(document, entity, sent['id'], attributes = attributes)
            tmp['ents'] += conv_ents

    tmp['text'] = document['annotations'][0]['value']
    tmp['uuid'] = 0
    
     
    options = {"colors" : {}}
    i = 0
    for entity in entities: 
        options['colors'][entity.upper()] = palette[i]
        i += 1

    displacy.render(tmp, manual=True, style = 'ent', options = options , jupyter=True)

