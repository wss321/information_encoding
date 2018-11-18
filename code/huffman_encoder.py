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
        self.codding = self.encode()
        self.source_entropy = self.entropy()
        self.mean_length = self.mean_code_length()
        self.encode_efficiency = self.encode_rate()

    def reduce(self):
        """缩减信源"""
        df = self.df.copy()
        S = []
        while len(df.index) > 1:
            df, s = reduce_info_src(df)
            S.append(s)
        d = dict()
        for i in S:
            # print(i)
            d = {**d, **i}
        return d

    def encode(self):
        """编码"""
        keys = []
        values = []
        codding = {}
        for key, value in self.reduce_pro.items():
            keys.append(key)
            values.append(value)
        for index, sym in enumerate(self.symbols):
            in_list = find_and_sort(sym, keys)
            # print(f'{sym}:{in_list}')
            code = ''
            for i in in_list:
                idx = keys.index(i)
                code += values[idx]
            codding[sym] = code
        keys = []
        values = []
        for key, value in codding.items():
            keys.append(key)
            values.append(value)
        codding = pd.DataFrame(values, index=keys, columns=['code'])
        return codding

    def mean_code_length(self):
        p = self.frequency.copy()
        code = self.codding.copy()
        df = pd.concat([p, code], axis=1)
        df['l_i'] = df.apply(lambda x: x['frequency'] * len(x['code']), axis=1)
        return df['l_i'].sum()

    def entropy(self):
        p = self.frequency.copy().values
        for idx in range(len(p)):
            if p[idx] != 0:
                p[idx] = p[idx] * np.log2(p[idx])
        return - p.sum()

    def encode_rate(self):
        return self.source_entropy / self.mean_length


if __name__ == '__main__':
    encoder = HuffmanEncoder(filename='../data/GameOfThrones.txt')
    print(encoder.codding)
    print(f'信源熵:{encoder.source_entropy:.2f}')
    print(f'平均码长:{encoder.mean_length:.2f}')
    print(f'编码效率:{encoder.encode_efficiency*100:.2f}%')
