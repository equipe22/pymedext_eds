#!/usr/bin/env python3

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
    app.run(host='0.0.0.0', port = 6666, debug=True)
