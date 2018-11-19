# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class HuffmanEncoder(object):
    """赫夫曼编码"""

    def __init__(self, filename='../data/GameOfThrones.txt'):
        self.filename = filename
        self.content = self.__load()
        self.num_and_freq = self.__count_sym_frequency()
        self.frequency = self.num_and_freq[['frequency']]
        self.number = self.num_and_freq[['number']]
        self.symbols = self.num_and_freq.index.tolist()
        self.reduce_pro = self.__reduce()
        self.code_word = self.__code_word()
        self.source_entropy = self.__entropy()
        self.mean_length = self.__mean_code_length()
        self.encode_efficiency = self.__encode_rate()

    def __load(self):
        """加载文件"""
        with open(self.filename, 'r', encoding='gbk') as f:
            lines = f.readlines()
        return ''.join(lines)

    def __count_sym_num(self):
        """计算每个符号数量"""
        sym = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)] + [' ']
        sym_num = np.zeros(len(sym), dtype=np.int64)
        for letter in self.content:
            if letter in sym:
                sym_num[sym.index(letter)] += 1

        df = pd.DataFrame(sym_num, index=sym)
        df.columns = ['number']
        return df

    def __count_sym_frequency(self):
        """计算符号频率"""
        df = self.__count_sym_num()
        df = df.sort_values(by='number', ascending=False)
        number = df[['number']].values
        number = number / number.sum()
        df['frequency'] = np.squeeze(number)
        return df

    def __reduce(self):
        """计算缩减信源"""

        def __reduce_info_src(data_frame):
            """一次缩减信源"""
            data_frame = data_frame.sort_values(by='frequency', ascending=False)
            end_2 = data_frame.tail(2)
            S = end_2.index.tolist()
            end_2 = end_2.copy()
            end_2.loc[''.join(S)] = end_2.apply(lambda x: x.sum())
            data_frame.drop(data_frame.index[[data_frame.index.tolist().index(i) for i in S]], inplace=True)
            data_frame = pd.concat([data_frame, end_2.tail(1)])
            return data_frame, {S[0]: '1', S[1]: '0'}

        df = self.num_and_freq.copy()
        s = []
        while len(df.index) > 1:
            df, s_i = __reduce_info_src(df)
            s.append(s_i)
        d = dict()
        for i in s:
            # print(i)
            d = {**d, **i}
        return d

    def __code_word(self):
        """编码成码字"""

        def __find_and_sort(symbol, string_list):
            """找到string_list中的包含符号symbol的所有字符串并按照长短排序"""
            sorted_list = []
            for string in string_list:
                if symbol in string:
                    sorted_list.append(string)
            sorted_list.sort(key=lambda i: len(i), reverse=True)
            return sorted_list

        keys = []
        values = []
        codding = {}
        for key, value in self.reduce_pro.items():
            keys.append(key)
            values.append(value)
        for index, sym in enumerate(self.symbols):
            in_list = __find_and_sort(sym, keys)
            # print(f'{sym}:{sorted_list}')
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
        codding = pd.DataFrame(values, index=keys, columns=['code_word'])
        return codding

    def __mean_code_length(self):
        """计算平均码长"""
        p = self.frequency.copy()
        code = self.code_word.copy()
        df = pd.concat([p, code], axis=1)
        df['l_i'] = df.apply(lambda x: x['frequency'] * len(x['code_word']), axis=1)
        return df['l_i'].sum()

    def __entropy(self):
        """计算信息熵"""
        p = self.frequency.copy().values
        for idx in range(len(p)):
            if p[idx] != 0:
                p[idx] = p[idx] * np.log2(p[idx])
        return - p.sum()

    def __encode_rate(self):
        """计算编码效率"""
        return self.source_entropy / self.mean_length


if __name__ == '__main__':
    encoder = HuffmanEncoder(filename='../data/GameOfThrones.txt')
    print(encoder.code_word)
    print(f'信源熵:{encoder.source_entropy:.2f}')
    print(f'平均码长:{encoder.mean_length:.2f}')
    print(f'编码效率:{encoder.encode_efficiency*100:.2f}%')
