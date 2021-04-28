
from flask import Flask, send_from_directory, request
import random
from pprint import pprint

from pymedext_eds.annotators import Endlines, SentenceTokenizer, Hypothesis, \
                                    ATCDFamille, SyntagmeTokenizer, Negation, RegexMatcher, \
                                    Pipeline, QuickUMLSAnnotator
from pymedext_eds.viz import display_annotations
from pymedextcore.document import Document
from pymedext_eds.utils import get_version_git

endlines = Endlines(['raw_text'], 'endlines', get_version_git('EndLines'))
sentences = SentenceTokenizer(['endlines'], 'sentence', get_version_git('SentenceTokenizer'))
#hypothesis = Hypothesis(['sentence'], 'hypothesis', 'hypothesis:v1')
family = ATCDFamille(['sentence'], 'context', get_version_git('ATCDFamille'))
syntagmes = SyntagmeTokenizer(['sentence'], 'syntagme', get_version_git('SyntagmeTokenizer'))
negation = Negation(['syntagme'], 'negation', get_version_git(' Negation'))
regex = RegexMatcher(['endlines','syntagme'], 'regex', get_version_git('RegexMatcher'), 'list_regexp.json')
# umls = QuickUMLSAnnotator(['syntagme'], 'umls', 'QuickUMLS:2020AB',
#                           quickumls_fp='data/quickumls_files/',
#                             overlapping_criteria='length',
#                             threshold=0.9,
#                             similarity_name='jaccard',
#                             window=5)

pipeline = Pipeline(pipeline = [endlines, sentences,  family, syntagmes, negation, regex])

app = Flask(__name__)

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)

@app.route('/annotate',methods = ['POST'])
def result():
    if request.method == 'POST':
        res = pipeline(request)
        docs = [Document.from_dict(doc) for doc in res['result'] ]
        return {'html': display_annotations(docs[0], ['regex'],
                attributes = ['context','negation'], jupyter=False ),
                'json' : docs[0].to_dict()}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type = int, help="Port", default = 5000)
    parser.add_argument("--debug", help="debug mode", action="store_true" )

    args = parser.parse_args()
    app.run(port=args.port,
            debug=args.debug)
