# RecSys模型论文复现-TF2
- 参考论文：https://arxiv.org/pdf/1706.06978.pdf
- 参考代码：https://github.com/StephenBo-China/recommendation_system_sort_model
- 原始数据集下载：https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

注：模型使用数据需要对原始数据集进行特征工程构建，本文序列化特征构建与参考作者方法思路不同，特征结果不同。

### model
- 模型文件夹
>- 目前仅上传DIN模型
- 简介
>- 论文输入特征包括：用户画像特征、用户行为序列特征以及候选Item特征；
>- 本文输入特征包括：用户画像特征、用户历史行为序列特征以及候选Item特征：
>>- user_profile_dict：用户画像特征；
>>- hist_behavior_dict：用户历史行为特征，行为序列中包含Item的category以及brand；
>>- target_item_dict：候选Item特征，与用户历史行为序列中Item的特征一致。
- 文章Attention机制说明
>- 输入为用户历史行为序列Item（Item_1） & 候选Item（Item_2）
>- 计算方式：Item_1 + Item_2 + (Item_1 - Item_2) + (Item_1 * Item_2)
>- 返回值：如论文所述，未使用softmax
- weight sum pooling机制
>- 目的：通过计算序列Item和候选Item，获取用户的兴趣模型，使得最终得到的用户embedding包含了兴趣的体现
>- 方法：(Attention score * Item_1)

