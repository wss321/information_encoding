# information_encoding
python 实现*信源编码*:
```
(1). Huffman 编码
(2). Shannon 编码
(3). Fano 编码
```
# 0.环境
```
python>=3.6
numpy
pandas
```
# 1. 数据
data/GameOfThrones.txt<br/>
# 2. 编码
```
运行 code/merge.py 
选择相应的编码
```
## (1). Huffman 编码
| 字符 | 码字 |
|:---:|:---------:|
| 空格 | 00 |
| e | 010 |
| t | 1010 |
| a | 1001 |
| o | 1000 |
| h | 0111 |
| n | 0110 |
| r | 11111 |
| s | 11110 |
| i | 11100 |
| d | 11011 |
| l | 11000 |
| w | 110101 |
| u | 110100 |
| m | 110011 |
| g | 101111 |
| y | 101110 |
| f | 101101 |
| c | 1110111 |
| b | 1110110 |
| k | 1100101 |
| p | 1100100 |
| v | 11101010 |
| T | 10110010 |
| I | 111010111 |
| S | 111010110 |
| H | 111010010 |
| A | 101100010 |
| L | 101100000 |
| W | 1110100111 |
| B | 1110100011 |
| M | 1110100010 |
| J | 1110100001 |
| R | 1110100000 |
| N | 1011001111 |
| D | 1011001101 |
| C | 1011000110 |
| Y | 1011000010 |
| q | 11101001100 |
| G | 10110011101 |
| K | 10110011100 |
| O | 10110011001 |
| F | 10110011000 |
| j | 10110001111 |
| P | 10110000111 |
| E | 10110000110 |
| z | 111010011011 |
| x | 111010011010 |
| V | 101100011101 |
| U | 1011000111001 |
| Q | 10110001110001 |
| X | 101100011100001 |
```
信源熵:4.22
平均码长:4.26
编码效率:99.03%
码字长度方差:2.98
```

## (2). Shannon 编码
| 字符 | 码字 |
|:---:|:---------:|
| 空格 | 000 |
| e | 0011 |
| t | 0100 |
| a | 01011 |
| o | 01101 |
| h | 01111 |
| n | 10001 |
| r | 10010 |
| s | 10100 |
| i | 10110 |
| d | 10111 |
| l | 11000 |
| w | 110100 |
| u | 110101 |
| m | 110110 |
| g | 110111 |
| y | 111000 |
| f | 111001 |
| c | 1110101 |
| b | 1110111 |
| k | 1111000 |
| p | 1111010 |
| v | 11110110 |
| T | 111101111 |
| I | 111110001 |
| S | 111110011 |
| H | 111110100 |
| A | 1111101100 |
| L | 1111101110 |
| W | 1111110000 |
| B | 1111110001 |
| M | 1111110010 |
| J | 1111110100 |
| R | 1111110101 |
| N | 1111110110 |
| D | 1111110111 |
| C | 11111110001 |
| Y | 11111110011 |
| q | 11111110100 |
| G | 11111110110 |
| K | 11111110111 |
| O | 11111111000 |
| F | 11111111001 |
| j | 111111110101 |
| P | 111111110111 |
| E | 111111111001 |
| z | 111111111010 |
| x | 111111111100 |
| V | 111111111110 |
| U | 11111111111101 |
| Q | 11111111111110 |
| X | 1111111111111111110 |
```
信源熵:4.22
平均码长:4.80
编码效率:87.88%
码字长度方差:2.05
```

## (3). Fano 编码
| 字符 | 码字 |
|:---:|:---------:|
| 空格 | 00 |
| e | 010 |
| t | 0110 |
| a | 01110 |
| o | 01111 |
| h | 1000 |
| n | 1001 |
| r | 1010 |
| s | 10110 |
| i | 10111 |
| d | 1100 |
| l | 11010 |
| w | 110110 |
| u | 110111 |
| m | 11100 |
| g | 111010 |
| y | 1110110 |
| f | 1110111 |
| c | 111100 |
| b | 1111010 |
| k | 1111011 |
| p | 1111100 |
| v | 11111010 |
| T | 111110110 |
| I | 111110111 |
| S | 111111000 |
| H | 111111001 |
| A | 111111010 |
| L | 1111110110 |
| W | 11111101110 |
| B | 11111101111 |
| M | 1111111000 |
| J | 1111111001 |
| R | 1111111010 |
| N | 11111110110 |
| D | 11111110111 |
| C | 11111111000 |
| Y | 11111111001 |
| q | 11111111010 |
| G | 111111110110 |
| K | 111111110111 |
| O | 11111111100 |
| F | 111111111010 |
| j | 111111111011 |
| P | 111111111100 |
| E | 111111111101 |
| z | 111111111110 |
| x | 1111111111110 |
| V | 11111111111110 |
| U | 111111111111110 |
| Q | 1111111111111110 |
| X | 11111111111111110 |
```
信源熵:4.22
平均码长:4.31
编码效率:97.95%
码字长度方差:3.23
```