import analysis
import baseline
import feature
import recall
import sort

import sys

def main():
    print(f"{sys.argv[0]}")
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            #   调用基线函数作为对比
            if sys.argv[i] == 'baseline':
                baseline
            #   分析数据集中的数据
            if sys.argv[i] == 'analysis':
                analysis 
    recall  #   召回任务
    feature #   特征工程
    sort    #   排序模型



if __name__ == '__main__' :
    main()