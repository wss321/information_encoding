# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def load(filename):
    with open(filename, 'r', encoding='gbk') as f:
        lines = f.readlines()
    return ''.join(lines)


def count_sym_num(content):
    """计算每个符号数量"""
    sym = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)] + [' ']
    sym_num = np.zeros(len(sym), dtype=np.int64)
    for letter in content:
        if letter in sym:
            sym_num[sym.index(letter)] += 1

    df = pd.DataFrame(sym_num, index=sym)
    df.columns = ['number']
    return df


def count_sym_frequency(content):
    """计算符号频率"""
    df = count_sym_num(content)
    df = df.sort_values(by='number', ascending=False)
    number = df[['number']].values
    number = number / number.sum()
    df['frequency'] = np.squeeze(number)
    return df


def reduce_info_src(df):
    """缩减一次信源"""
    df = df.sort_values(by='frequency', ascending=False)
    end_2 = df.tail(2)
    S = end_2.index.tolist()
    end_2 = end_2.copy()
    end_2.loc[''.join(S)] = end_2.apply(lambda x: x.sum())
    df.drop(df.index[[df.index.tolist().index(i) for i in S]], inplace=True)
    df = pd.concat([df, end_2.tail(1)])
    return df, {S[0]: '1', S[1]: '0'}


def find_and_sort(sym, data):
    in_list = []
    for i in data:
        if sym in i:
            in_list.append(i)
    in_list.sort(key=lambda i: len(i), reverse=True)
    return in_list


class HuffmanEncoder(object):
    """赫夫曼编码"""

    def __init__(self, filename='../data/GameOfThrones.txt'):
        self.filename = filename
        self.content = load(self.filename)
        self.df = count_sym_frequency(self.content)
        self.frequency = self.df[['frequency']]
        self.number = self.df[['number']]
        self.symbols = self.df.index.tolist()
        self.reduce_pro = self.reduce()

    def reduce(self):
        """缩减信源"""
        df = self.df
        S = []
        while len(df.index) > 1:
            df, s = reduce_info_src(df)
            S.append(s)
        d = dict()
        for i in S:
            d = {**d, **i}
        return d

    def encode(self):
        """编码"""
        keys = []
        values = []
        coding = {}
        for key, value in self.reduce_pro.items():
            keys.append(key)
            values.append(value)
        for index, sym in enumerate(encoder.symbols):
            in_list = find_and_sort(sym, keys)
            code = ''
            for i in in_list[:-1]:
                idx = keys.index(i)
                code += values[idx]
            coding[sym] = code
        return coding


if __name__ == '__main__':
    encoder = HuffmanEncoder(filename='../data/GameOfThrones.txt')
    codding = encoder.encode()
    for key, value in codding.items():
        print(f'{key}:{value}')
