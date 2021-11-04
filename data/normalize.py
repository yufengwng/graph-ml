#!/usr/bin/env python3

import numpy as np
import pandas as pd


cites = pd.read_csv('cora/cites.csv')
labels = pd.read_csv('cora/paper.csv')
content = pd.read_csv('cora/content.csv')

labels.sort_values('paper_id', inplace=True)
papers = labels['paper_id']

unique_words = content['word_cited_id'].unique()
unique_words.sort()

rows = []
for id in papers:
    matches = content[content['paper_id'] == id]
    mentions = set(matches['word_cited_id'])
    row = [id]
    for word in unique_words:
        row.append(1 if word in mentions else 0)
    rows.append(row)
columns = ['paper_id'] + list(unique_words)
nodes = pd.DataFrame(np.array(rows), columns=columns)

edges = cites.reindex(columns=['citing_paper_id', 'cited_paper_id'])
edges.sort_values('citing_paper_id', inplace=True)

nodes.to_csv('cora-nodes.csv', index=False)
edges.to_csv('cora-edges.csv', index=False)
labels.to_csv('cora-labels.csv', index=False)
