Ceci est un *fork* du repo de l'équipe 22. Pour une présentation succinte, voir la [fiche du wiki d'équipe](https://gitlab.eds.aphp.fr/equipedatascience/equipedatascience-notes/-/wikis/PyMedExt/).


# Pymedext annotators for the EDS pipeline

## Installation

Requires the installation of PyMedExt_core [PyMedExt_core](https://github.com/equipe22/pymedext_core)
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
from pymedext_eds.utils import rawtext_loader

from pymedext_eds.annotators import Endlines, SentenceTokenizer, \
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
- SectionSplitter:
    - Segments the text into sections
    - output: Annotations
- SentenceTokenizer:
    - Tokenize the text in sentences
    - input: cleaned text from Endlines or sections
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
    - Extracts medical concepts from UMLS using [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS)
    - output: Annotations
- MedicationAnnotator:
    - Extracts medications informations using a deep learning pipeline
    - output: Annotations


### QuickUMLS installation (copied from [Georgetown-IR-Lab/QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS))

Installation

1. Obtain a UMLS installation This tool requires you to have a valid UMLS installation on disk. To install UMLS, you must first obtain a [license](https://uts.nlm.nih.gov/license.html) from the National Library of Medicine; then you should download all UMLS files from [this page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html); finally, you can install UMLS using the [MetamorphoSys](https://www.nlm.nih.gov/pubs/factsheets/umlsmetamorph.html) tool as explained in [this guide](https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html). The installation can be removed once the system has been initialized.
2. Install QuickUMLS: You can do so by either running `pip install quickuml`s or `python setup.py install`. On macOS, using anaconda is **strongly recommended**†.
3. Create a QuickUMLS installation Initialize the system by running `python -m quickumls.install <umls_installation_path> <destination_path>`, where `<umls_installation_path>` is where the installation files are (in particular, we need `MRCONSO.RRF` and `MRSTY.RRF`) and `<destination_path>` is the directory where the QuickUmls data files should be installed. This process will take between 5 and 30 minutes depending how fast the CPU and the drive where UMLS and QuickUMLS files are stored are (on a system with a Intel i7 6700K CPU and a 7200 RPM hard drive, initialization takes 8.5 minutes). 

 `python -m quickumls.install` supports the following optional arguments:
- -L / --lowercase: if used, all concept terms are folded to lowercase before being processed. This option typically increases recall, but it might reduce precision;
- -U / --normalize-unicode: if used, expressions with non-ASCII characters are converted to the closest combination of ASCII characters.
- -E / --language: Specify the language to consider for UMLS concepts; by default, English is used. For a complete list of languages, please see [this table provided by NLM](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#LAT).
- -d / --database-backend: Specify which database backend to use for QuickUMLS. The two options are leveldb and unqlite. The latter supports multi-process reading and has better unicode compatibility, and it used as default for all new 1.4 installations; the former is still used as default when instantiating a QuickUMLS client. More info about differences between the two databases and migration info are available [here](https://github.com/Georgetown-IR-Lab/QuickUMLS/wiki/Migration-QuickUMLS-1.3-to-1.4).

†: If the installation fails on macOS when using Anaconda, install leveldb first by running `conda install -c conda-forge python-leveldb`.

## Run a simple server

### Define the server and the pipeline:

```python
import flask

from flask import Flask, render_template, request

from pymedext_eds.annotators import Endlines, SentenceTokenizer, Hypothesis, \
                                    ATCDFamille, SyntagmeTokenizer, Negation, RegexMatcher, \
                                    Pipeline

endlines = Endlines(['raw_text'], 'endlines', 'endlines:v1')
sentences = SentenceTokenizer(['endlines'], 'sentence', 'sentenceTokenizer:v1')
hypothesis = Hypothesis(['sentence'], 'hypothesis', 'hypothesis:v1')
family = ATCDFamille(['sentence'], 'context', 'ATCDfamily:v1')
syntagmes = SyntagmeTokenizer(['sentence'], 'syntagme', 'SyntagmeTokenizer:v1')
negation = Negation(['syntagme'], 'negation', 'Negation:v1')
regex = RegexMatcher(['endlines','syntagme'], 'regex', 'RegexMatcher:v1', 'list_regexp.json')

pipeline = Pipeline(pipeline = [endlines, sentences, hypothesis, family, syntagmes, negation, regex])

app=Flask(__name__)

@app.route('/annotate',methods = ['POST'])
def result():
    if request.method == 'POST':

        return pipeline.__call__(request)

if __name__ == '__main__':
    app.run(port = 6666, debug=True)
```

Save this code in `demo_flask_server.py` and run it using:

```bash
python demo_flask_server.py
```

### Query the server:

```python
import requests
from pymedextcore.document import Document

data_path = pkg_resources.resource_filename('pymedext_eds', 'data/demo')
file_list = glob(data_path + '/*.txt')
docs = [rawtext_loader(x) for x in file_list]

json_doc = [doc.to_dict() for doc in docs]
res =  requests.post(f"http://127.0.0.1:6666/annotate", json = json_doc)
if res.status_code == 200:
    res = res.json()['result']
    docs = [Document.from_dict(doc) for doc in res ]
```

## Run a docker server

### define the git credentials
first create a file .git-credentials and replace user and pass by your
github credentials such has

``` bash
https://user:pass@github.com
```

WARNING :never add it on the git !!!

### build the images

```bash

docker build -f eds_apps/Dockerfile_backend -t pymedext-eds:v1 .


#if proxy add
docker build -f eds_apps/Dockerfile_backend -t pymedext-eds:v1 \
--buildargs http_proxy="proxy" \
--buildargs https_proxy="proxy" .


```

### start the backend server

``` bash

docker run --rm  -d -p 6666:6666 pymedext-eds:v1 python3 demo_flask.py

```

