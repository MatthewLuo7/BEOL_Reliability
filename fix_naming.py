import os
import re

# pattern 1：包含 vm 的旧格式
pattern_with_vm = re.compile(
    r"vm(?P<vm>[\d\.]+)_ll(?P<ll>[\d\.]+)_vx(?P<vx>[\d\.]+)_vy(?P<vy>[\d\.]+)_vz(?P<vz>[\d\.]+)_ly(?P<ly>[\d\.]+)_lz(?P<lz>[\d\.]+)_r(?P<r>[\d\.]+)"
)

# pattern 2：不包含 vm 的旧格式
pattern_no_vm = re.compile(
    r"ll(?P<ll>[\d\.]+)_vx(?P<vx>[\d\.]+)_vy(?P<vy>[\d\.]+)_vz(?P<vz>[\d\.]+)_ly(?P<ly>[\d\.]+)_lz(?P<lz>[\d\.]+)_r(?P<r>[\d\.]+)"
)

def rename_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)

            # -------- 情况1：带 vm --------
            match = pattern_with_vm.fullmatch(dirname)
            if match:
                g = match.groupdict()
                new_name = (
                    f"vm{float(g['vm']):.2f}_"
                    f"ll{float(g['ll']):.2f}_"
                    f"vy{float(g['vy']):.2f}_"
                    f"vz{float(g['vz']):.2f}_"
                    f"lx{float(g['vx']):.2f}_"
                    f"ly{float(g['ly']):.2f}_"
                    f"lz{float(g['lz']):.2f}_"
                    f"r{float(g['r']):.2f}"
                )

            else:
                # -------- 情况2：不带 vm --------
                match = pattern_no_vm.fullmatch(dirname)
                if match:
                    g = match.groupdict()
                    new_name = (
                        f"ll{float(g['ll']):.2f}_"
                        f"vy{float(g['vy']):.2f}_"
                        f"vz{float(g['vz']):.2f}_"
                        f"lx{float(g['vx']):.2f}_"
                        f"ly{float(g['ly']):.2f}_"
                        f"lz{float(g['lz']):.2f}_"
                        f"r{float(g['r']):.2f}"
                    )
                else:
                    continue  # 不匹配任何格式，跳过

            new_path = os.path.join(dirpath, new_name)

            if old_path == new_path:
                continue

            if os.path.exists(new_path):
                print(f"[跳过] 目标已存在: {new_path}")
                continue

            print(f"[重命名] {old_path} -> {new_path}")
            os.rename(old_path, new_path)


if __name__ == "__main__":
    root = input("请输入根目录路径: ").strip()
    rename_folders(root)