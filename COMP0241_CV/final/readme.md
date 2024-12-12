### 文件说明
- utils.py: 读取数据
- functions.py: 我用到的各种函数都放里面了，包括 detect_largest_circle 我也复制了一份放里面
- Task2.py & Task3.py
- run_tasks.py: 运行 task2 和 task3 各任务

Task1 的代码我没放进去

### 数据集格式说明：

- Segmentation_Testset: task1 的数据集，我没动
- dataset: 我们自己的数据集：
  - dual_cam: 双目摄像机拍摄的图片。由视频拆分成图片，并统一成 19000 张
    - 2: 我们在 2 楼倾斜拍摄的图片
    - 6: 在 6 楼水平拍摄的图片
    - G1 & G2: 在 G 层垂直拍摄的图片，由于中间缺了一部分，所以分成 2 份，删除了开头遮挡部分，文件名为时间戳。
  - video: 我们自己录制的原始视频
