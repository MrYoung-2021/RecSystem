# 导入相关包
# %matplotlib inline
import gc
import os
import re
import sys
import warnings
from pathlib import Path
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

# plt.rc('font', size=13)

warnings.filterwarnings("ignore")

from funrec.utils import load_env_with_fallback

load_env_with_fallback()

RAW_DATA_PATH = Path('D:\\Workspace\\RecSystem')
PROCESSED_DATA_PATH = Path('D:\\Workspace\\RecSystem')

data_path = RAW_DATA_PATH / 'data'

# 训练集
trn_click = pd.read_csv(data_path / 'train_click_log.csv')
item_df = pd.read_csv(data_path / 'articles.csv')
item_df = item_df.rename(columns={'article_id': 'click_article_id'})  #重命名，方便后续match
item_emb_df = pd.read_csv(data_path / 'articles_emb.csv')

# 测试集
tst_click = pd.read_csv(data_path / 'testA_click_log.csv')

# 对每个用户的点击时间戳进行排序
trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)

#计算用户点击文章的次数，并添加新的一列count
trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')

trn_click = trn_click.merge(item_df, how='left', on=['click_article_id'])
trn_click.head()

#用户点击日志信息
trn_click.info()

trn_click.describe()

#训练集中的用户数量为20w
trn_click.user_id.nunique()

trn_click.groupby('user_id')['click_article_id'].count().min()  # 训练集里面每个用户至少点击了两篇文章

tst_click = tst_click.merge(item_df, how='left', on=['click_article_id'])
tst_click.head()

tst_click.describe()

#测试集中的用户数量为5w
tst_click.user_id.nunique()

#新闻文章数据集浏览
item_df.head()

item_df['words_count'].value_counts()

print(item_df['category_id'].nunique())     # 461个文章主题
_ = item_df['category_id'].hist(figsize=(5, 4), grid=False)

item_df.shape       # 364047篇文章

item_emb_df.head()

item_emb_df.shape

user_click_merge = pd.concat([trn_click, tst_click])

#用户重复点击
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
user_click_count[:5]

user_click_count[user_click_count['count']>7]

user_click_count['count'].unique()

#用户点击新闻次数
user_click_count.loc[:,'count'].value_counts()



# 分析用户点击环境变化是否明显，这里随机采样10个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=5, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 'click_region','click_referrer_type']

user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count().values, reverse=True)
item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)

tmp = user_click_merge.sort_values('click_timestamp')
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1))
union_item = tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
union_item[['count']].describe()
user_click_merge.groupby('user_id')['category_id'].nunique().reset_index().describe()

#更加详细的参数
user_click_merge.groupby('user_id')['words_count'].mean().reset_index().describe()

#为了更好的可视化，这里把时间进行归一化操作
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])

user_click_merge = user_click_merge.sort_values('click_timestamp')

user_click_merge.head()

def mean_diff_time_func(df, col):
    df = pd.DataFrame(df, columns=[col])
    df['time_shift1'] = df[col].shift(1).fillna(0)
    df['diff_time'] = abs(df[col] - df['time_shift1'])
    return df['diff_time'].mean()

# 点击时间差的平均值
mean_diff_click_time = user_click_merge.groupby('user_id')[['click_timestamp', 'created_at_ts']].apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))

# 前后点击文章的创建时间差的平均值
mean_diff_created_time = user_click_merge.groupby('user_id')[['click_timestamp', 'created_at_ts']].apply(lambda x: mean_diff_time_func(x, 'created_at_ts'))

# 用户前后点击文章的相似性分布
item_idx_2_rawid_dict = dict(zip(item_emb_df['article_id'], item_emb_df.index))
del item_emb_df['article_id']
item_emb_np = np.ascontiguousarray(item_emb_df.values, dtype=np.float32)

# 随机选择5个用户，查看这些用户前后查看文章的相似性
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

sub_user_info.head()

def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_emb_np[item_idx_2_rawid_dict[item_list[i]]]
        emb2 = item_emb_np[item_idx_2_rawid_dict[item_list[i+1]]]
        sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))
    sim_list.append(0)
    return sim_list

# plt.figure(figsize=(5, 3))
for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    # plt.plot(item_sim_list)