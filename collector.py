import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

URL = 'https://attack.mitre.org/'
sess = requests.Session()
res = sess.get(URL)
bs = BeautifulSoup(res.text)

tactic_row = bs.find('table', 'matrix').find_all('td', 'tactic name')
technique_table = bs.find_all('table', 'techniques-table')
matrix = []
label_list = []
name_list = []

# Convert HTML into matrix
for tactic_column, td in zip(technique_table, tactic_row):
    tactic_label = td.a['title']
    tactic_name = td.a.text
    tactic = {
        'label': tactic_label,
        'name': tactic_name
    }
    techniques = []
    _label_list = [tactic_label]
    _name_list = [tactic_name]
    
    technique_rows = tactic_column.find_all('tr', 'technique-row')
    for row in technique_rows:
        technique = {}
        
        tech = row.find('div', 'technique-cell')
        tech_label = tech.a['title']
        tech_name = tech.a.text.split('\xa0')[0]
        technique['label'] = tech_label
        technique['name'] = tech_name
        _label_list.append(tech_label)
        _name_list.append(tech_name)
        
        subtechniques = row.find_all('div', 'subtechnique')
        subtechnique = []
        if subtechniques:
            for subtech in subtechniques:
                subtech_label = subtech.a['title']
                subtech_name = subtech.a.text
                subtechnique.append({
                    'label': subtech_label,
                    'name': subtech_name
                })
                _label_list.append(subtech_label)
                _name_list.append(subtech_name)
        technique['subtechniques'] = subtechnique
        techniques.append(technique)
    label_list.append(_label_list)
    name_list.append(_name_list)
    tactic['techniques'] = techniques
    matrix.append(tactic)

# Techniques list
tac_list = [l[0] for l in label_list]
tech_list = []
tech_per_ta_list = [l[1:] for l in label_list]
for te in tech_per_ta_list:
    tech_list.extend(te)
tech_list = list(set(tech_list))

# Technique name-id mapping
technique_map = {}
for tactic in matrix:
    for technique in tactic['techniques']:
        technique_map[technique['name']] = technique['label']
        if technique['subtechniques']:
            for subtechnique in technique['subtechniques']:
                technique_map[subtechnique['name']] = subtechnique['label']

# Collect all text
from nltk.tokenize import sent_tokenize
URL = 'https://attack.mitre.org/techniques/'
rows = []
pattern_cite = '\[\d*\]'
pattern_abbr = '(etc|e\.g|i\.e)\.'

sess = requests.Session()
for techniques in tqdm(label_list):
    for technique in tqdm(techniques[1:]):
        tactic = techniques[0]
        if '.' in technique:
            _technique = '/'.join(technique.split('.'))
        else:
            _technique = technique
        res = sess.get(URL + _technique, proxies=proxy)
        bs = BeautifulSoup(res.text)
        for p in bs.find('div', 'description-body').find_all('p'):
            p = re.sub(pattern_cite, '', p.text)
            p = re.sub('ex\.', 'e.g.', p)
            p = re.sub(pattern_abbr, '', p)
            sents = sent_tokenize(p)
            for sent in sents:
                if sent:
                    rows.append([sent.strip(), tactic, technique])
        for l in bs.find('div', 'description-body').find_all('li'):
            l = re.sub(pattern_cite, '', l.text)
            l = re.sub(pattern_abbr, '', l)
            if l:
                rows.append([l.strip(), tactic, technique])
        if bs.find('h2', id='examples') or bs.find('h2', id='Mitigations'):
            for table in bs.find_all('table', 'table table-bordered table-alternate mt-2'):
                for p in table.find_all('p'):
                    p = re.sub(pattern_cite, '', p.text)
                    p = re.sub(pattern_abbr, '', p)
                    sents = sent_tokenize(p)
                    for sent in sents:
                        if sent:
                            rows.append([sent.strip(), tactic, technique])
        try:
            if bs.find('h2', id='detection'):
                for p in bs.find('h2', id='detection').next_sibling.next_sibling.find_all('p'):
                    p = re.sub(pattern_cite, '', p.text)
                    p = re.sub('ex\.', 'e.g.', p)
                    p = re.sub(pattern_abbr, '', p)
                    sents = sent_tokenize(p)
                    for sent in sents:
                        if sent:
                            rows.append([sent.strip(), tactic, technique])
        except AttributeError:
            print(_technique)
        # print('Extracted' + ' ' + technique)
with open('attack_original.csv', 'w', encoding='utf-8', newline='') as f:
    f = csv.writer(f)
    f.writerow(['text', 'tactic', 'technique'])
    f.writerows(rows)