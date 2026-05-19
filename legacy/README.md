# Legacy Shell Scripts

这个目录保存旧版完整 shell 流程，仅用于追溯历史运行方式。

当前推荐入口：

- 批量实验：`python3 run_experiment.py configs/experiments/*.yaml`
- 单次运行：`python3 run_pipeline.py ...`
- 兼容薄包装：`run_all_test.sh`、`run_kitti_raw_city_2011_09_26_drive_0001.sh`

新增实验请优先写到 `configs/experiments/`，不要继续扩展这里的脚本。
