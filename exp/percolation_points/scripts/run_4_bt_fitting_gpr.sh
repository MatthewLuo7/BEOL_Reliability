vm_offset_list="0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5"
m_list="1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0"

# via2line, r=0.35
python -m exp.percolation_points.via2line_bt_fitting_gpr -r 0.35 --vm-offset-list $vm_offset_list --m-list $m_list --radius-N 2.0 --ll-space 10.5 --via-dim-x 10.5 --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-y 21.0 --line-dim-z 21.0 --cpu-num 64 --chunk-size 4
# via2line, r=0.45
python -m exp.percolation_points.via2line_bt_fitting_gpr -r 0.45 --vm-offset-list $vm_offset_list --m-list $m_list --radius-N 2.0 --ll-space 10.5 --via-dim-x 10.5 --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-y 21.0 --line-dim-z 21.0 --cpu-num 64 --chunk-size 4
# via2line, r=0.55
python -m exp.percolation_points.via2line_bt_fitting_gpr -r 0.55 --vm-offset-list $vm_offset_list --m-list $m_list --radius-N 2.0 --ll-space 10.5 --via-dim-x 10.5 --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-y 21.0 --line-dim-z 21.0 --cpu-num 64 --chunk-size 4
# via2line, r=0.65
python -m exp.percolation_points.via2line_bt_fitting_gpr -r 0.65 --vm-offset-list $vm_offset_list --m-list $m_list --radius-N 2.0 --ll-space 10.5 --via-dim-x 10.5 --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-y 21.0 --line-dim-z 21.0 --cpu-num 64 --chunk-size 4
# via2line, r=0.75
python -m exp.percolation_points.via2line_bt_fitting_gpr -r 0.75 --vm-offset-list $vm_offset_list --m-list $m_list --radius-N 2.0 --ll-space 10.5 --via-dim-x 10.5 --via-dim-y 10.5 --via-dim-z 21.0 --line-dim-y 21.0 --line-dim-z 21.0 --cpu-num 64 --chunk-size 4
