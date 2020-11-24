
from flask import Flask, send_from_directory, request
import random
from pprint import pprint

from pymedext_eds.annotators import Endlines, SentenceTokenizer, Hypothesis, \
                                    ATCDFamille, SyntagmeTokenizer, Negation, RegexMatcher, \
                                    Pipeline, QuickUMLSAnnotator
from pymedext_eds.viz import display_annotations
from pymedextcore.document import Document

endlines = Endlines(['raw_text'], 'endlines', 'endlines:v1')
sentences = SentenceTokenizer(['endlines'], 'sentence', 'sentenceTokenizer:v1')
#hypothesis = Hypothesis(['sentence'], 'hypothesis', 'hypothesis:v1')
family = ATCDFamille(['sentence'], 'context', 'ATCDfamily:v1')
syntagmes = SyntagmeTokenizer(['sentence'], 'syntagme', 'SyntagmeTokenizer:v1')
negation = Negation(['syntagme'], 'negation', 'Negation:v1')
regex = RegexMatcher(['endlines','syntagme'], 'regex', 'RegexMatcher:v1', 'list_regexp.json')
umls = QuickUMLSAnnotator(['syntagme'], 'umls', 'QuickUMLS:2020AB',
                          quickumls_fp='data/quickumls_files/',
                            overlapping_criteria='length',
                            threshold=0.9,
                            similarity_name='jaccard',
                            window=5)

pipeline = Pipeline(pipeline = [endlines, sentences,  family, syntagmes, negation, regex, umls])

app = Flask(__name__)

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client/public', 'index.html')

@app.route('/annotate',methods = ['POST'])
def result():
    if request.method == 'POST':

        res = pipeline(request)
        docs = [Document.from_dict(doc) for doc in res['result'] ]
        pprint([x.to_dict() for x in docs[0].get_annotations('regex')])
        return {'html': display_annotations(docs[0], ['umls','regex'], attributes = ['context','negation'], jupyter=False ), 'json' : docs[0].to_dict()}

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)

@app.route("/rand")
def hello():
    return str(random.randint(0, 100))

if __name__ == "__main__":
    app.run(debug=True)
