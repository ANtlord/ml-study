#!/usr/bin/env python
import operator
from collections import defaultdict
from typing import List

import pandas
from pandas import DataFrame
import re

WOMEN_SALUTATIONS = [
    'Mrs', 'Miss', 'Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dr'
]


def parse_women_names(values: List[str]):
    name_points = defaultdict(int)
    for item in values:
        res = re.match(r'[^,]*, (%s)\. ' % '|'.join(WOMEN_SALUTATIONS), item)
        if res is None:
            raise Exception(item)

        item_mut = item[res.end():]
        res = re.search('\([^\)]*\)', item_mut)
        if res is not None:
            item_mut = item_mut[res.start() + 1:res.end() - 1]
            if item_mut.startswith('"Mrs '):
                item_mut = item_mut[5:-1]

        name = item_mut.split()[0]
        name_points[name] += 1
    return max(name_points.items(), key=operator.itemgetter(1))[0]


def answer(value, filename):
    value = str(value)
    with open(filename, 'w') as f:
        f.write(value)
    print(value)

if __name__ == '__main__':
    data = pandas.read_csv('train.csv', index_col='PassengerId')  # type: DataFrame
    gender_col = data['Sex']
    counts = gender_col.value_counts()
    answer('%s %s' % (counts['male'], counts['female']), 'first.txt')

    surv_col = data['Survived']
    answer(round(surv_col.value_counts()[1] / len(data) * 100, 2), 'second.txt')

    pclass_col = data['Pclass']
    answer(round(pclass_col.value_counts()[1] / len(data) * 100, 2), 'third.txt')

    age_col = data['Age']
    answer('%s %s' % (age_col.mean(), age_col.median()), 'fourth.txt')

    sibl_col = data['SibSp']
    parch_col = data['Parch']
    answer(round(sibl_col.corr(parch_col, 'pearson'), 2), 'fifth.txt')

    sibl_col = data['Name']
    women = data[data['Sex'] == 'female']
    answer(parse_women_names([x for x in women['Name']]), 'sixth.txt')
