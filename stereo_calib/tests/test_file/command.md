编译
cmake --build stereo_calib/build -j4

跑单次匹配

python3 stereo_calib/scripts/superpoint_stereo_match.py   --dataset_mode project   --img_dir /home/hello/pml/mycalib/stereo_calib/tests/test_file/   --weights stereo_calib/scripts/superpoint_v1.pth   --output /home/hello/pml/mycalib/stereo_calib/tests/test_file/matches.json

跑单次BA

./stereo_calib/build/bin/run_stereo_calib   --input /home/hello/pml/mycalib/stereo_calib/tests/test_file/matches.json   --output /home/hello/pml/mycalib/stereo_calib/tests/test_file/result.json   --max_iter 100   --max_reproj_error 10
