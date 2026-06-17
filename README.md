# 项目目录说明

```python
cs_agent
├─ README.md
├─ answer_general_query.py #处理通识类问题的代码
├─ answer_product_query.py # 处理产品类问题的代码
├─ catalog # 分别对中英文手册生成的目录内容
│  ├─ chinese_handbook_catalog.json
│  └─ english_handbook_catalog.json
├─ chunk.py # 手册切块、入库的代码
├─ config.py # 配置文件，只需配置MILVUS_COLLECTION_NAME_DEFAULT变量即可，设置向量数据库的集合名称
├─ data # 原始数据
├─ del_bank_by_handbook.py # 根据手册名删除向量数据库中该手册的所有数据
├─ generate_catalog.py # 生成手册目录的代码
├─ generate_handbook_name.py # 生成英文手册名称的代码
├─ handbook_name_gemini.json # 存放生成的英文手册名称
├─ interface.py # 接口代码，提供给前端调用
├─ llm_judge_result.py # 使用LLM给最终的结果打分的代码
├─ milvus-docker-compose.yml # 部署milvus向量数据库的docker-compose文件
├─ pipline.py #  智能体的入口文件
├─ preprocess.py # 预处理代码，使用LLM对中英文手册内容进行预处理
├─ processed_data # 预处理之后的数据，其中_formatted.txt为后缀的文件会用于接下来的切块和入库
├─ prompts.py # 存放一些提示词
├─ pyproject.toml 
├─ query_classification.py # 对问题进行分类，预测手册名称的代码
├─ question_public.csv # 公开的测试问题
├─ retriever.py # 检索代码，根据问题检索相关的手册内容
├─ submission # 存放提交结果的文件夹
├─ submit.py # 根据测试问题生成运行pipeline，生成最终的提交结果
├─ test_ensembles_query_classification.json # 查询分类的测试结果
├─ utils.py # 工具函数
|- milvus-backup-files #向量数据库的备份
|- backup.yaml #恢复备份的时候需要用到的配置文件
|- mc #minio的命令行工具
|- milvus-backup # milvus的备份和恢复的工具
└─ uv.lock
```



# 依赖安装

1. 下载uv

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. 进行项目根目录，执行下面的命令下载依赖

   ```shell
   uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
   source .venv/bin/activate
   ```

3. 配置key,按文件中的注释要求配置即可

   ```shell
   cp .env.example .env
   ```

4. 配置.env 和 config.py中的内容，config.py中只要配置MILVUS_COLLECTION_NAME_DEFAULT变量即可，设置向量数据库的集合名称

5. 启动milvus数据库，

   ```shell
   sudo docker compose -f milvus-docker-compose.yml up -d
   ```

# 使用已经有的向量数据库复现

1. 可以选择重新建立向量数据库，也可以使用已经建立好的向量数据库

   重新建立向量数据库，需要运行`chunk.py`,对预处理好的数据进行切块并将pic转为对应的描述，再入库

   如果需要使用我建立的向量数据库，则需要在启动之后将数据导入进去，`milvus-backup-files`中就是对应的数据,只需要在启动了milvus数据库了之后运行下面的命令即可将数据导入向量数据库中即可

```shell
./mc alias set dst http://127.0.0.1:9000 minioadmin minioadmin
./mc mb dst/a-bucket --ignore-existing
./mc cp --recursive \
  ./milvus-backup-files/milvus_to_aliyun_20260615 \
  dst/a-bucket/backup/
  
./milvus-backup restore \
  -n milvus_to_aliyun_20260615 \
  --config backup.yaml
```

2. 启动接口

   ```shell
   python interface.py
   ```

   访问`http://localhost:8000/scalar` 查看接口文档



# 从0开始复现

1. 运行`preprocess.py`使用`gemini-2.5-pro`对手册预处理

2. 之后运行`chunk.py`对文本进行切块、将pic转为文本描述，最后将chunk向量化并入库

3. 运行`generate_handbook_name.py`生成英文的手册名称

4. 运行`generate_catalog.py`生成手册的目录

5. 启动接口

   ```shell
   python interface.py
   ```

   访问`http://localhost:8000/scalar` 查看接口文档
