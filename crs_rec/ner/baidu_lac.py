"""
https://github.com/baidu/lac
"""

import argparse

from LAC import LAC


def debug():
    # lac = LAC(mode='seg')
    lac = LAC(mode='lac')

    # 单个样本输入，输入为Unicode编码的字符串
    text = u"小度你给我讲小度小度说一遍"
    text = '你喜欢谁	我喜欢威霖'
    seg_result = lac.run(text)
    print(seg_result)


def demo():
    # 装载分词模型
    lac = LAC(mode='seg')

    # 单个样本输入，输入为Unicode编码的字符串
    text = u"LAC是个优秀的分词工具"
    seg_result = lac.run(text)

    # 批量样本输入, 输入为多个句子组成的list，平均速率会更快
    texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
    seg_result = lac.run(texts)
    print(seg_result)


def main():
    """
        参数说明
        1：语料库路径
        2：模型保存路径
    """
    parser = argparse.ArgumentParser(description="参数解析")
    # 训练相关参数
    parser.add_argument("--train", action='store_true', help='重新训练')
    parser.add_argument("--export", action='store_true', help='导出结果')
    parser.add_argument("--debug", action='store_true', help='调试')
    #
    args, argv = parser.parse_known_args()
    print('* args: ', vars(args))
    print("* unknown args: {}".format(argv))
    #
    if args.train:
        pass
    elif args.debug:
        debug()
    else:
        debug()


if __name__ == '__main__':
    main()
