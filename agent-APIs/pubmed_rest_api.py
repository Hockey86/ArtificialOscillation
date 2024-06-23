import datetime
import requests


def get_pubmed_OA_manuscript_fulltext(pmid):
    try:
        pmid = int(pmid)
        rq = requests.get(f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode')
        rq.raise_for_status()
        rq = rq.json()
    except Exception as ee:
        return None

    n_passage = len(rq['documents'][0]['passages'])
    section_types = [rq['documents'][0]['passages'][x]['infons']['section_type'] for x in range(n_passage)]
    section_type_types = [rq['documents'][0]['passages'][x]['infons']['type'] for x in range(n_passage)]
    texts = [rq['documents'][0]['passages'][x]['text'] for x in range(n_passage)]

    date = datetime.datetime.strptime(rq['date'], '%Y%m%d')
    title = [t for t,s in zip(texts, section_types) if s=='TITLE']
    assert len(title)==1; title = title[0]
    abstract = '\n'.join([t for t,s in zip(texts, section_types) if s=='ABSTRACT'])
    abstract = '\n'.join([t for t,s in zip(texts, section_types) if s=='ABSTRACT'])
    paragraphs = [t for t,s,st in zip(texts, section_types, section_type_types) if s not in ['TITLE', 'ABSTRACT', 'REF', 'ABBR', 'COMP_INT'] and st=='paragraph']

    return {'title':title, 'abstract':abstract, 'paragraphs':paragraphs, 'date':date}

