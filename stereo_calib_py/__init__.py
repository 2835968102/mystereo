"""Python helpers for the stereo calibration pipeline.

这个包承接原来 `run_pipeline.py` 里的非 CLI 逻辑：
路径解析、KITTI 标定转换、子命令执行，以及单次 pipeline 编排。
顶层脚本保留给命令行入口，这里保留给可复用业务逻辑。
"""
