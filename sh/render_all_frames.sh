# 循环范围
for i in {20..70}; do
  # 使用sed命令将config.txt中的idx参数修改为i
  sed -i "s/render_idx = .*/render_idx = $i/" ./configs/test_kid-running.txt
  python render_monocular_bt.py --config ./configs/test_kid-running.txt
done