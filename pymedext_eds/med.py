import datetime
from typing import List

import flair
import ray
import torch
from ray.serve.utils import _get_logger
from torch.utils.data import Dataset

# from .extract.extractor import  ToFlair, collate_text, Extractor
# from .extract.preprocess import TextPreprocessing
# from .extract.postprocess import ConllPostProcess

logger = _get_logger()

import re
# import torch
# import flair
# from typing import List
from pymedextcore.document import Document
from pymedextcore.annotators import Annotator, Annotation
from flair.models import SequenceTagger
from flair.data import Sentence

from .pyromedi.romedi import Romedi
from .constants import CLASS_NORM
from .normalisation import clean_drug, clean_class, clean_freq, clean_dose


class TextDataset(Dataset):
    def __init__(self, observations, to_flair, transform=None):
        self.transform = transform
        self.to_flair = to_flair
        self.observations = observations

    def __getitem__(self, index):
        # Load label
        observation = self.observations[index]

        try:
            assert self.transform and (observation["note_text"]) and (observation["note_text"].strip() != "")
            observation["conll"], observation["note_text"] = self.transform.transform_text(observation["note_text"])
            observation["flair"] = self.to_flair(observation["conll"])
            observation['n_sent'] = len(observation["flair"])
            observation['preprocessing_status'] = True
        except:
            observation['preprocessing_status'] = False

        return observation

    def __len__(self):
        return len(self.observations) - 1


# from pymedextcore.document import Document
from .annotators import Endlines, SentenceTokenizer, SectionSplitter
# from pymedext_eds.med import MedicationAnnotator, MedicationNormalizer
import pkg_resources
from glob import glob


class Pipeline:
    """
    Pipeline designed to work with Ray 1.2.0
    """
    
    def __init__(self, device=None, mini_batch_size=128):

        if device is None:
            if ray.get_gpu_ids():
                device = f'cuda'
            else:
                device = 'cpu'

        self.endlines = Endlines(["raw_text"], "clean_text", ID="endlines")
        self.sections = SectionSplitter(['clean_text'], "section", ID='sections')
        self.sentenceSplitter = SentenceTokenizer(["section"], "sentence", ID="sentences")

        self.models_param = [
            {
                'tagger_path': 'data/models/apmed5/entities/final-model.pt',
                'tag_name': 'entity_pred'
            },
            {
                'tagger_path': 'data/models/apmed5/events/final-model.pt',
                'tag_name': 'event_pred'
            },
            {
                'tagger_path': "data/models/apmed5/drugblob/final-model.pt",
                'tag_name': 'drugblob_pred'
            },
        ]

        self.med = MedicationAnnotator(['sentence'], 'med', ID='med:v2', models_param=self.models_param, device=device)

        data_path = pkg_resources.resource_filename('pymedext_eds', 'data/romedi')
        romedi_path = glob(data_path + '/*.p')[0]

        self.norm = MedicationNormalizer(['ENT/DRUG', 'ENT/CLASS'], 'normalized_mention', ID='norm',
                                         romedi_path=romedi_path)

        self.pipeline = [self.endlines, self.sections, self.sentenceSplitter, self.med, self.norm]

    def process(self, payload):
        """
        Does the heavy lifting. Pools sentences together for faster inference.
        """
        
        docs = {doc['ID']: Document.from_dict(doc) for doc in payload}
        
        for doc in docs.values():
            doc.annotate([self.endlines, self.sections, self.sentenceSplitter])
        
        sentences = []
        for doc in docs.values():
            for annotation in doc.get_annotations('sentence'):
                annotation.attributes['doc_id'] = doc.ID
                sentences.append(annotation)
                
        logger.info(f"Processing {len(docs)} documents, {len(sentences)} sentences")
            
        placeholder_doc = Document('')
        placeholder_doc.annotations = sentences
        
        placeholder_doc.annotate([self.med, self.norm])
        
        for annotation in placeholder_doc.annotations:
            if annotation.type != 'sentence':
                doc = docs[annotation.attributes['doc_id']]
                del annotation.attributes['doc_id']
                doc.annotations.append(annotation)
        
        return [doc.to_dict() for doc in docs.values()]

    async def __call__(self, request):
        """
        Adaptation to Ray 1.2, with async starlette request mechanism.
        """

        payload = await request.json()
        res = self.process(payload)

        return {'result': res}


class MedicationAnnotator(Annotator):
    def __init__(self, key_input, key_output, ID,
                 models_param,
                 mini_batch_size=128,
                 max_sentence_length=1024,
                 device="cpu"):

        super().__init__(key_input, key_output, ID)

        self.mini_batch_size = mini_batch_size
        flair.device = torch.device(device)
        
        self.max_sentence_length = max_sentence_length

        self.model_zoo = []
        self.tagged_name = []
        for i, param in enumerate(models_param):
            tagger = SequenceTagger.load(param["tagger_path"]).to(flair.device)
            tagger.tag_type = param["tag_name"]
            self.tagged_name.append(param["tag_name"])
            
            # for elmo
            if hasattr(tagger.embeddings.embeddings[0], "ee"):
                tagger.embeddings.embeddings[0].ee.cuda_device = flair.device.index

            # get terminology tag if any
            add_tags = []
            for e in tagger.embeddings.embeddings:
                if hasattr(e, "field"):
                    field = getattr(e, "field")
                    add_tags.append(field)
                    
                # And manually set the batch size
                if hasattr(e, "batch_size"):
                    e.batch_size = 512

            self.model_zoo.append(
                (param["tag_name"], tagger, add_tags)
            )

    def infer_flair(self, sentences):
        flair_sentences = []
        for s in sentences:
            # "Bug" in the tokenisation : if the sentence is too long (> 512 tokens)
            # we get a runtime error.
            if len(s.value) <= self.max_sentence_length:
                flair_sentences.append(Sentence(s.value, start_position=s.span[0]))
            else:
                flair_sentences.append(Sentence(s.value[:self.max_sentence_length], start_position=s.span[0]))
                # TODO : add sentences by splitting along spaces ?
                # for i in range(len(s.value) // self.max_sentence_length):
                #     flair_sentences.append(
                #         Sentence(
                #             s.value[i * self.max_sentence_length:(i + 1) * self.max_sentence_length], 
                #             start_position=s.span[0] + i * self.max_sentence_length
                #         )
                #     )

        flair_sentences = self._filter_empty_sentences(flair_sentences)

        # for each mini_batch
        for sent_i in range(0, len(flair_sentences), self.mini_batch_size):
            mini_batch = flair_sentences[sent_i:sent_i + self.mini_batch_size]

            # infer each model
            for tag_name, model, add_tags in self.model_zoo:
                mini_batch = self.infer_minibatch(model, mini_batch)

                # clear embeddings
            #               store_embeddings(mini_batch, 'gpu')
            for sentence in mini_batch:
                sentence.clear_embeddings()

            flair_sentences[sent_i:sent_i + self.mini_batch_size] = mini_batch

        return flair_sentences

    def infer_minibatch(self, model, mini_batch):
        with torch.no_grad():
            if model.use_crf:
                transitions = model.transitions.detach().cpu().numpy()
            else:
                transitions = None

            feature: torch.Tensor = model.forward(mini_batch)
            tags, all_tags = model._obtain_labels(
                feature=feature,
                batch_sentences=mini_batch,
                transitions=transitions,
                get_all_tags=False,
            )

            for (sentence, sent_tags) in zip(mini_batch, tags):
                for (token, tag) in zip(sentence.tokens, sent_tags):
                    token.add_tag_label(model.tag_type, tag)

        return mini_batch

    @staticmethod
    def _get_sentence_entities(flair_sentence, source):

        res = []

        for entity in flair_sentence.get_spans():

            label = entity.get_labels()[0]

            # filter out events
            if label.value[0:3] != 'EV/':
                res.append({'value': entity.text,
                            'span': (
                            entity.start_pos + flair_sentence.start_pos, entity.end_pos + flair_sentence.start_pos),
                            'type': label.value,
                            'attributes': {'score': label.score},
                            'source_ID': source.ID})
        return res

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            logger.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    def _postprocess_entities(self, sentences, flair_sentences):
        entities = []

        for sentence in flair_sentences:
            # print(f"{sentence.to_original_text()}, {sentence.start_pos}-{sentence.end_pos}")
            if not sentence.get_spans():
                continue

            source = [s for s in sentences if s.span[0] == sentence.start_pos][0]
            entities.append(self._get_sentence_entities(sentence, source))

        return entities

    def _tag_sentences(self, sentences):

        flair_sentences = []
        for s in sentences:
            flair_sentences.append(Sentence(s.value, start_position=s.span[0]))

        self.multi.predict(flair_sentences)

        return flair_sentences

    def annotate_function(self, _input):

        sentences = self.get_all_key_input(_input)
        flair_sentences = self.infer_flair(sentences)
        annotated_sentences = self._postprocess_entities(sentences, flair_sentences)
        res = []

        if len(annotated_sentences) == 0:
            return res

        for sent in annotated_sentences:

            if len(sent) == 0:
                continue

            sent_att = [x.attributes.copy() for x in sentences if x.ID == sent[0]['source_ID']][0]

            # if only 1 entity in sent => append Annotation
            if len(sent) == 1:
                if sent[0]['type'] in ['ENT/DRUG', 'ENT/CLASS']:
                    res.append(Annotation(
                        type=sent[0]['type'],
                        value=sent[0]['value'],
                        span=sent[0]['span'],
                        source=self.ID,
                        source_ID=sent[0]['source_ID'],
                        attributes={**sent_att, **sent[0]['attributes']}
                    ))

                    # pprint('---- drug not blob ----')
                    # pprint (sent[0])

            # else:
            #     pprint('---- isolated not drug ----')
            #     pprint (sent[0])

            # if more than 1 entity
            else:

                new_sent = []

                drug_blobs = [x for x in sent if x['type'] == 'REL/DRUG_BLOB']

                # if drug_blobs => combine all entities in drub_blob
                if len(drug_blobs) > 0:

                    blob_ents = [[] for blob in drug_blobs]

                    for ent in sent:
                        ent['in_blob'] = False
                        for i, blob in enumerate(drug_blobs):
                            blob_start = blob['span'][0]
                            blob_end = blob['span'][1]
                            if ent['span'][0] >= blob_start and ent['span'][1] <= blob_end:
                                ent['in_blob'] = True
                                blob_ents[i].append(ent)

                        # if entity not in blob => append Annotation (if ENT/DRUG or ENT_CLASS)
                        if not ent['in_blob']:

                            if ent['type'] in ['ENT/DRUG', 'ENT/CLASS']:
                                res.append(Annotation(
                                    type=ent['type'],
                                    value=ent['value'],
                                    span=ent['span'],
                                    source=self.ID,
                                    source_ID=ent['source_ID'],
                                    attributes={**sent_att, **ent['attributes']}
                                ))

                                # pprint('---- drug not blob ----')
                            # pprint (ent)

                        # else:

                        #     pprint('---- not blob and not drug ----')
                        #     pprint(ent)

                    # in each blob combine drug and class with their attributes
                    for blob in blob_ents:
                        drugs_class = [x for x in blob if x['type'] in ['ENT/DRUG', 'ENT/CLASS']]
                        att_list = [x for x in blob if x['type'] not in ['ENT/DRUG', 'ENT/CLASS', 'REL/DRUG_BLOB']]
                        snippet = [x['value'] for x in blob if x['type'] == 'REL/DRUG_BLOB'][0]

                        att_reformat = {}
                        for att in att_list:
                            if att['type'] not in att_reformat.keys():
                                att_reformat[att['type']] = []
                            att_reformat[att['type']] += [att]

                        # combined = []

                        for drug in drugs_class:
                            drug['attributes'] = {**sent_att, **drug['attributes'], **att_reformat}
                            drug['attributes']['snippet'] = snippet

                            # combined.append(drug)

                            res.append(Annotation(
                                type=drug['type'],
                                value=drug['value'],
                                span=drug['span'],
                                source=self.ID,
                                source_ID=drug['source_ID'],
                                attributes=drug['attributes']
                            ))

                            # pprint('---- in blob ----')
                        # pprint(combined)



                else:
                    # keep drug and class only
                    # pprint('---- other -----')

                    for ent in sent:
                        if ent['type'] in ['ENT/DRUG', 'ENT/CLASS']:
                            res.append(Annotation(
                                type=ent['type'],
                                value=ent['value'],
                                span=ent['span'],
                                source=self.ID,
                                source_ID=ent['source_ID'],
                                attributes={**sent_att, **ent['attributes']}
                            ))
        return res

    @staticmethod
    def doc_to_omop(annotated_doc, new_norm=False):

        annots = [x.to_dict() for x in
                  annotated_doc.get_annotations('ENT/DRUG') + annotated_doc.get_annotations('ENT/CLASS')]
        sentences = [x.to_dict() for x in annotated_doc.get_annotations('sentence')]
        if annots == []:
            return []

        note_id = annotated_doc.source_ID
        person_id = annotated_doc.attributes.get('person_id')

        res = []

        for drug in annots:

            section = drug["attributes"]['section']
            sentence = [x['value'] for x in sentences if x['ID'] == drug['source_ID']][0]

            norm = None
            if 'normalized_mention' in drug["attributes"].keys():
                if (drug['attributes']['normalized_mention'] != []) & (drug['attributes']['normalized_mention'] != {}):
                    if drug['type'] == 'ENT/DRUG':
                        norm = drug['attributes']['normalized_mention']['ATC7']
                    else:
                        norm = drug['attributes']['normalized_mention']

                    # new normalization with concept code separated from label 
                    # deactivated by default to ensure retro-compatibility
                    if norm is not None and new_norm:
                        norm_extract = re.search('^([A-Z0-9]+) \((.+)\)', norm)
                        if norm_extract:
                            norm = norm_extract.group(1)
                            drug['attributes']['ATC_LABEL'] = norm_extract.group(2)
            dose = None
            dose_norm = None
            route = None
            duration = None
            duration_norm = None
            freq = None
            freq_norm = None

            modifiers = []
            if drug['attributes'] != {}:
                for att, val in drug['attributes'].items():

                    if att in ['ENT/ROUTE', 'ENT/DOSE', 'ENT/DURATION', 'ENT/FREQ', 'ENT/CONDITION']:
                        for v in val:
                            modifiers.append(f"{att}='{v['value']}'")
                            if 'normalized_value' in v.keys():
                                modifiers.append(f"{att}_norm='{v['normalized_mention']}'")

                    elif att != "normalized_mention":
                        modifiers.append(f"{att}='{val}'")

            note_nlp_item = {
                'note_nlp_id': None,
                'note_id': note_id,
                'person_id': person_id,
                'section_concept_id': section,
                'snippet': sentence,
                'offset_begin': drug['span'][0],
                'offset_end': drug['span'][1],
                'lexical_variant': drug['value'],
                'note_nlp_concept_id': norm,
                'note_nlp_source_concept_id': 'ATC',
                'nlp_system': "medext_v3",
                'nlp_date': f"{datetime.datetime.today():%Y-%m-%d}",
                'nlp_datetime': f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                'term_exists': None,
                'term_temporal': None,
                'term_modifiers': ','.join(modifiers),
                'validation_level_id': 'automated'
            }

            res.append(note_nlp_item)

        return res


class MedicationNormalizer(Annotator):
    def __init__(self, key_input, key_output, ID, romedi_path, class_norm=CLASS_NORM):

        super().__init__(key_input, key_output, ID)

        self.romedi = Romedi(from_cache=romedi_path)
        self.class_norm = class_norm

        self.cache_mentions = {"ENT/DRUG": {},
                               "ENT/CLASS": {},
                               "ENT/FREQ": {},
                               "ENT/DOSE": {}}

    def normalize(self, ent_type, mention):
        if mention.lower() in self.cache_mentions[ent_type].keys():
            return self.cache_mentions[ent_type][mention.lower()]
        else:
            if ent_type == 'ENT/DRUG':
                cleaned = clean_drug(mention, self.romedi)
            elif ent_type == 'ENT/CLASS':
                cleaned = clean_class(mention, self.class_norm)
            elif ent_type == 'ENT/FREQ':
                cleaned = clean_freq(mention)
            elif ent_type == 'ENT/DOSE':
                cleaned = clean_dose(mention)

            self.cache_mentions[ent_type][mention.lower()] = cleaned
            return cleaned

    def annotate_function(self, _input):

        inps = self.get_all_key_input(_input)

        for annot in inps:
            annot.attributes[self.key_output] = self.normalize(annot.type, annot.value)

            if 'ENT/DOSE' in annot.attributes.keys() or 'ENT/FREQ' in annot.attributes.keys():
                for k, v in annot.attributes.items():
                    if k == 'ENT/DOSE':
                        for att in v:
                            att[self.key_output] = self.normalize('ENT/DOSE', att['value'])
                    elif k == 'ENT/FREQ':
                        for att in v:
                            att[self.key_output] = self.normalize('ENT/FREQ', att['value'])
