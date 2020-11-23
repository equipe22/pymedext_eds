# Pymedext annotators for the EDS pipeline 

## Installation 

Requires the installation of PyMedExt_core [https://github.com/equipe22/pymedext_core](PyMedExt_core)
It can be done using `requirements.txt`

```bash
pip install -r requirements.txt
```

Installation via pip: 

```bash
pip install git+git://github.com/equipe22/pymedext_eds.git@master#egg=pymedext_eds
```

Cloning the repository:

```bash
git clone https://github.com/equipe22/pymedext_eds.git
cd pymedext_eds
pip install .
```

## Basic usage

All the annotators are defined in the pymedext_eds.annotators module. You will find a description of the existing annotators in the next section.

- First, import the annotators and text :

```python
from pymedext_eds.annotators import rawtext_loader, Endlines, SentenceTokenizer, \
                                    RegexMatcher, Pipeline 

from pymedext_eds.viz import display_annotations
```

- Load documents: 

```python
data_path = pkg_resources.resource_filename('pymedext_eds', 'data/demo')
file_list = glob(data_path + '/*.txt')
docs = [rawtext_loader(x) for x in file_list]
```

- Declare the pipeline: 

```python
endlines = Endlines(['raw_text'], 'endlines', 'endlines:v1')
sentences = SentenceTokenizer(['endlines'], 'sentence', 'sentenceTokenizer:v1')
regex = RegexMatcher(['endlines','syntagme'], 'regex', 'RegexMatcher:v1', 'list_regexp.json')

pipeline = Pipeline(pipeline = [endlines, sentences, regex])
```

- Use the pipeline to annotate:

```python
annotated_docs = pipeline.annotate(docs)
```

- Explore annotations by type :

```python
from pprint import pprint
pprint(annotated_docs[0].get_annotations('regex')[10].to_dict())
```

- Display annotations in text (using displacy)

```python
display_annotations(chunk[0], ['regex'])
```




## Existing annotators

- Endlines: 
    - Used to clean the text when using text extracted from PDFs. Removes erroneous endlines introduced by pdf to text conversion. 
    - input : raw_text
    - output: Annotations
- SentenceTokenizer: 
    - Tokenize the text in sentences
    - input: cleaned text from Endlines
    - output: Annotations
- Hypothesis: 
    - Classification of sentences regarding the degree of certainty
    - input: sentences 
    - output: Attributes
- ATCDFamille: 
    - Classification of sentences regarding the subject (patient or family)
    - input: sentences 
    - output: Attributes
- SyntagmeTokenizer: 
    - Segmentation of sentences into syntagms
    - input: sentences
    - output: Annotations
- Negation: 
    - Classification of syntagms according to the polarity
    - input: syntagm
    - output: Attributes
- RegexMatcher: 
    - Extracts informations using predefined regexs
    - input: sentence or syntagm
    - output: Annotations
- QuickUMLSAnnotator: 
    - Extracts medical concepts from UMLS using QuickUMLS
    - output: Annotations
- MedicationAnnotator:
    - Extracts medications informations using a deep learning pipeline
    - output: Annotations 


