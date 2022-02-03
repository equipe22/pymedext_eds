
from flask import Flask, send_from_directory, request
import random
from pprint import pprint

from pymedext_eds.annotators import Endlines, SentenceTokenizer, Hypothesis, \
                                    ATCDFamille, SyntagmeTokenizer, Negation, RegexMatcher, \
                                    Pipeline, QuickUMLSAnnotator, RuSHSentenceTokenizer
from pymedext_eds.viz import display_annotations
from pymedextcore.document import Document
from pymedext_eds.utils import get_version_git

sentences = RuSHSentenceTokenizer(['raw_text'], 'sentence', get_version_git('RuSHSentenceTokenizer'))
family = ATCDFamille(['sentence'], 'context', get_version_git('ATCDFamille'))
syntagmes = SyntagmeTokenizer(['sentence'], 'syntagme', get_version_git('SyntagmeTokenizer'))
negation = Negation(['syntagme'], 'negation', get_version_git(' Negation'))

app = Flask(__name__)

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client_regexp/build', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client_regexp/build', path)

@app.route('/annotate',methods = ['POST'])
def result():
    if request.method == 'POST':
        dt = request.get_json()

        regex = RegexMatcher(key_input=['sentence','syntagme'], 
                             key_output='regex',  
                             ID=get_version_git('RegexMatcher'), 
                             regexp_file=dt['options']['regex'])
        pipeline = Pipeline(pipeline = [sentences,  family, syntagmes, negation, regex])

        res = pipeline.process(dt['doc'])

        print(dt['options'])
        docs = [Document.from_dict(doc) for doc in res ]
        return {'html': display_annotations(docs[0], [dt['options']['type']],
                attributes = ['negation'], jupyter=False),
                'json' : docs[0].to_dict()}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type = int, help="Port", default = 5000)
    parser.add_argument("--debug", help="debug mode", action="store_true" )

    args = parser.parse_args()
    app.run(port=args.port,
            debug=args.debug)
