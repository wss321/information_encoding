# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class HuffmanEncoder(object):
    """赫夫曼编码"""

    def __init__(self, filename='../data/GameOfThrones.txt'):
        self.filename = filename
        self.content = self.__load()
        self.reduce_source = self.__reduce()
        self.encode_table = self.__encode()
        self.code_word = self.encode_table['code_word'].to_frame()
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
            data_frame.drop(S, inplace=True)
            data_frame = pd.concat([data_frame, end_2.tail(1)])
            return data_frame, {S[0]: '1', S[1]: '0'}

        num_and_freq = self.__count_sym_frequency()
        df = num_and_freq.copy()
        s = []
        while len(df.index) > 1:
            df, s_i = __reduce_info_src(df)
            s.append(s_i)
        d = dict()
        for i in s:
            d = {**d, **i}
        return d

    def __encode(self):
        """编码成码字"""

        def __find_and_sort(symbol, string_list):
            """找到string_list中的包含符号symbol的所有字符串并按照长短排序"""
            sorted_list = []
            for string in string_list:
                if symbol in string:
                    sorted_list.append(string)
            sorted_list.sort(key=lambda i: len(i), reverse=True)
            return sorted_list

        result = self.__count_sym_frequency()
        zero_freq_item = result[result['frequency'] == 0].index.tolist()
        result.drop(zero_freq_item, inplace=True)
        keys = []
        values = []
        code_word = {}
        for key, value in self.reduce_source.items():
            keys.append(key)
            values.append(value)
        for index, sym in enumerate(result.index.tolist()):
            in_list = __find_and_sort(sym, keys)
            # print(f'{sym}:{sorted_list}')
            code = ''
            for i in in_list:
                idx = keys.index(i)
                code += values[idx]
            code_word[sym] = code
        keys = []
        values = []
        code_len = []
        for key, value in code_word.items():
            keys.append(key)
            values.append(value)
            code_len.append(len(value))
        result['code_len'] = code_len
        result['code_word'] = values
        return result

    def __mean_code_length(self):
        """计算平均码长"""
        p = self.encode_table.copy()['frequency']
        code_len = self.encode_table.copy()['code_len']
        df = pd.concat([p, code_len], axis=1)
        df['l_i'] = df.apply(lambda x: x['frequency'] * x['code_len'], axis=1)
        return df['l_i'].sum()

    def __entropy(self):
        """计算信息熵"""
        p = self.encode_table.copy()['frequency'].values
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
