import re

def doc_to_target(doc):
    en2ga = {'science/technology': 'eolaíocht/teicneolaíocht', 'travel': 'taisteal', 'politics': 'polaitíocht', 'sports': 'spóirt', 'health': 'sláinte', 'entertainment': 'siamsaíocht', 'geography': 'tíreolaíocht'}
    return en2ga[doc['category']]
