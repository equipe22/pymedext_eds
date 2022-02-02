from spacy import displacy
import re

# def get_dict_annotations(document, _type, source_id = None, target_id = None):
#         res = []
#         for anno in document.annotations:
#             if source_id is not None:
#                 if anno.source_ID != source_id:
#                     continue
#             if target_id is not None:
#                 if anno.target_ID != target_id:
#                     continue
#             if anno.type == _type:
#                 res.append(anno)
#         return res

def convert_to_displacy(document, entity_type, attributes=None, label_key = 'label'):
    ents= []
    annots = document.get_annotations(entity_type)
    #annots = [x for x in document if x['type'] in entity_type and x['source_ID'] == source_id]
    if len(annots) > 0:
        annots = document.get_annotations(entity_type)
        ents= []
        for annot in annots:
            
            if (annot.attributes is not None) and (label_key in annot.attributes.keys()):
                label = annot.attributes[label_key]
            else:
                label = entity_type.upper()
            if attributes is not None:
                for att, val in annot.attributes.items():
                    if att in attributes:
                        label += f'/{val}'
            #label = entity_type.upper()
            ents.append({"start": annot.span[0], 'end':annot.span[1], "label": label})
            drug_id = annot.ID



    return sorted(ents, key = lambda i: i['start'])




def display_annotations(document,  entities = ["ENT/DRUG", "ENT/DOSE"], attributes = None,
                        text_source = 'raw_text',
                        palette = [ '#ffb3ba','#ffdfba','#ffffba','#baffc9','#bae1ff'],
                       label_key = 'label', jupyter = True):

    tmp = { "ents": []}

    for entity in entities:
        conv_ents = convert_to_displacy(document, entity, attributes = attributes, label_key = label_key)
        tmp['ents'] += conv_ents

    if text_source == 'raw_text':
        tmp['text'] = document.annotations[0].value
    else: 
        segments = [x.to_dict() for x in document.get_annotations(_type=text_source)]
        tmp['text'] = ''
        for segment in segments: 
            offset = len(tmp['text'])
            offset_diff = segment['span'][0] - offset
            if offset_diff > 0 : 
                tmp['text'] += ' ' * (offset_diff-1) + '\n'
            tmp['text'] += segment['value']
            
    tmp['uuid'] = 0
    
    tmp['ents'] = sorted(tmp['ents'], key = lambda i: i['start'])

    options = {"colors" : {}}
    i = 0
    for entity in entities:
        col = palette[i % len(palette)]
        options['colors'][entity.upper()] = col
        i += 1

    return displacy.render(tmp, manual=True, style = 'ent', options = options , jupyter=jupyter, minify=True)