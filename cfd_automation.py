import os
import subprocess
import time
import numpy as np
import pandas as pd
import shutil
from typing import List, Tuple


class CFDAutomation:
    def __init__(self):
        # 主工作目录
        self.work_dir = r'F:\essay_gt\opt'
        self.base_path = self.work_dir  # 保持一致性

        # 确保工作目录存在
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # 文件路径配置
        self.files = {
            'grid': os.path.join(self.base_path, 'varing_grid_fin_final.txt'),
            'new_data': os.path.join(self.base_path, 'data_new.txt'),
            'doe': os.path.join(self.base_path, 'doe_dynamic.txt'),
            'cl_output': os.path.join(self.base_path, 'cl-rfile.out'),
            'cd_output': os.path.join(self.base_path, 'cd-rfile.out'),
            'cm_output': os.path.join(self.base_path, 'cm-rfile.out')
        }

        # 脚本路径配置
        self.scripts = {
            'scdm': os.path.join(self.base_path, 'run_scdm.bat'),
            'icem': os.path.join(self.base_path, 'start_icem.bat'),
            'fluent': os.path.join(self.base_path, 'start_fluent.bat'),
            'matlab': os.path.join(self.base_path, 'run_matlab.bat')
        }

        # 需要监控和移动的文件列表
        self.monitor_files = [
            'test3.dat.h5',
            'tetra_cmd.log',
            'cd-rfile.out',
            'cl-rfile.out',
            'cm-rfile.out',
            'cn_rfile.out',
            'temp_tetra.tin',
            'test3.cas.h5'
        ]

    def _move_generated_files(self):
        """移动生成的文件到工作目录"""
        try:
            original_dir = os.getcwd()
            for file in self.monitor_files:
                source_path = os.path.join(original_dir, file)
                target_path = os.path.join(self.work_dir, file)

                if os.path.exists(source_path):
                    # 如果目标文件已存在，先删除
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    # 移动文件
                    shutil.move(source_path, target_path)
                    print(f"已将 {file} 从 {source_path} 移动到 {target_path}")
                else:
                    print(f"警告: 源文件不存在: {source_path}")

        except Exception as e:
            print(f"移动文件时出错: {str(e)}")
            # raise

    def modify_grid_file(self, chord: float, distance: float, fai: float):
        """修改网格文件中的参数"""
        try:
            with open(self.files['grid'], 'r') as f:
                lines = f.readlines()

            # 修改指定行的参数
            lines[15] = f"chord = {chord}\n"
            lines[20] = f"Distance_Final = {distance}\n"
            lines[23] = f"fai = {fai}\n"

            with open(self.files['grid'], 'w') as f:
                f.writelines(lines)

            print(f"网格文件参数更新成功：chord={chord}, distance={distance}, fai={fai}")

        except Exception as e:
            print(f"修改网格文件时出错: {str(e)}")
            raise

    def run_cfd_process(self, script_path: str, wait_time: int = 10) -> bool:
        """运行CFD相关进程并等待完成"""
        try:
            print(f"\n开始执行: {os.path.basename(script_path)}")
            process = subprocess.Popen(script_path, shell=True)
            process.wait()  # 等待进程完成

            # 如果是 Fluent 脚本，增加特殊处理
            if 'fluent' in script_path.lower():
                print("\n等待 Fluent 生成必要文件...")
                max_wait = 60  # 最大等待时间（秒）
                check_interval = 5  # 检查间隔（秒）
                waited = 0

                while waited < max_wait:
                    # 检查必要文件
                    cas_exists = os.path.exists(os.path.join(self.work_dir, 'test3.cas.h5'))
                    dat_exists = os.path.exists(os.path.join(self.work_dir, 'test3.dat.h5'))

                    print(f"\n检查文件状态 (等待时间: {waited}秒):")
                    print(f"test3.cas.h5: {'存在' if cas_exists else '不存在'}")
                    print(f"test3.dat.h5: {'存在' if dat_exists else '不存在'}")

                    if cas_exists and dat_exists:
                        print("\nFluent 所需文件已全部生成")
                        break

                    time.sleep(check_interval)
                    waited += check_interval

                if waited >= max_wait:
                    print("\n警告: Fluent 文件生成超时")
                    return False
            else:
                # 其他程序的正常等待时间
                time.sleep(wait_time)

            if process.returncode != 0:
                print(f"\n进程执行失败: {os.path.basename(script_path)}")
                return False

            print(f"\n成功完成: {os.path.basename(script_path)}")
            return True

        except Exception as e:
            print(f"\n执行{os.path.basename(script_path)}时出错: {str(e)}")
            return False



    """
    注意：此方法返回Cm的绝对值（正数）
    物理意义上，Cm为负值表示抬头力矩，为正值表示低头力矩
    但在优化中，我们关注的是力矩的大小而非方向，因此取绝对值
    """
    def read_cm_from_file(self, file_path: str) -> float:
        """
        从Cm-rfile.out文件读取最后100行第二列的平均值并取绝对值

        参数:
        ----
        file_path : str
            cm-rfile.out文件的路径

        返回:
        ----
        float : Cm绝对值的平均值（总是正值）
        """
        try:
            # 读取整个文件
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 确保文件至少有100行，否则使用所有可用行
            n_lines = min(100, len(lines))
            last_lines = lines[-n_lines:]

            # 从每行提取第二列的值
            cm_values = []
            for line in last_lines:
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保行有足够的列
                    try:
                        cm_value = float(parts[1])  # 第二列
                        cm_values.append(cm_value)
                    except ValueError:
                        # 忽略无法转换为浮点数的值
                        pass

            # 计算平均值并取绝对值
            if cm_values:
                avg_cm = sum(cm_values) / len(cm_values)
                abs_cm = abs(avg_cm)
                print(f"Cm文件分析: 使用了{len(cm_values)}行数据，原始平均值={avg_cm:.6f}，绝对值={abs_cm:.6f}")
                print(f"调试: Cm 原始平均值 = {avg_cm}, 绝对值 = {abs_cm}")
                return abs_cm  # 返回绝对值（正数）
            else:
                print("警告: cm-rfile.out文件中没有找到有效的Cm值")
                return 0.0

        except Exception as e:
            print(f"读取cm-rfile.out时出错: {str(e)}")
            # 如果出错，尝试使用常规方法读取最后一行并取绝对值
            try:
                with open(file_path, 'r') as f:
                    cm = float(f.readlines()[-1].split()[-1])
                abs_cm = abs(cm)
                print(f"使用备用方法读取Cm: 原始值={cm}，绝对值={abs_cm}")
                return abs_cm  # 返回绝对值
            except:
                print("备用方法也失败，返回0")
                return 0.0

    def read_output_values(self) -> Tuple[float, float, float]:
        """读取CFD计算结果"""
        try:
            # 读取各个输出文件的最后一行最后一列
            with open(self.files['cl_output'], 'r') as f:
                cl = float(f.readlines()[-1].split()[-1])

            with open(self.files['cd_output'], 'r') as f:
                cd = float(f.readlines()[-1].split()[-1])

            cm = self.read_cm_from_file(self.files['cm_output'])

            print(f"读取CFD结果：Cl={cl}, Cd={cd}, Cm={cm} (Cm已取绝对值)")
            return cl, cd, cm

        except Exception as e:
            print(f"读取输出文件时出错: {str(e)}")
            raise

    def update_data_files(self, cl: float, cd: float, cm: float):
        """更新数据文件中的因变量值"""
        try:
            # 读取当前的新数据文件
            data = pd.read_csv(self.files['new_data'], delimiter='\t')

            # 更新最后一行的因变量值
            data.loc[data.index[-1], ['Cl', 'Cd', 'Cm']] = [cl, cd, cm]

            # 保存更新后的数据
            data.to_csv(self.files['new_data'], sep='\t', index=False)

            print("已更新data_new.txt中的因变量值")

            # 运行MATLAB脚本将新数据添加到DOE文件
            if not self.run_cfd_process(self.scripts['matlab']):
                raise Exception("执行MATLAB脚本失败")

            print("已将新数据添加到doe_dynamic.txt")

        except Exception as e:
            print(f"更新数据文件时出错: {str(e)}")
            raise

    def run_cfd_workflow(self, point: np.ndarray) -> Tuple[float, float, float]:
        """执行完整的CFD工作流程"""
        try:
            # 保存当前工作目录
            original_dir = os.getcwd()

            try:
                # 切换到工作目录
                os.chdir(self.work_dir)
                print(f"\n当前工作目录: {os.getcwd()}")
                print(f"处理点: chord={point[0]:.6f}, distance={point[1]:.6f}, fai={point[2]:.6f}")

                # 1. 修改网格文件
                self.modify_grid_file(chord=point[0], distance=point[1], fai=point[2])

                # 2. 按顺序执行CFD工具
                # SCDM
                if not self.run_cfd_process(self.scripts['scdm'], wait_time=15):
                    raise Exception("SCDM执行失败")

                # ICEM
                if not self.run_cfd_process(self.scripts['icem'], wait_time=15):
                    raise Exception("ICEM执行失败")

                # Fluent (增加等待时间)
                if not self.run_cfd_process(self.scripts['fluent'], wait_time=30):
                    raise Exception("Fluent执行失败")

                # 最终检查所有必要文件
                required_files = [
                    'test3.cas.h5',
                    'test3.dat.h5',
                    'cl-rfile.out',
                    'cd-rfile.out',
                    'cm-rfile.out'
                ]

                missing_files = []
                for file in required_files:
                    file_path = os.path.join(self.work_dir, file)
                    if not os.path.exists(file_path):
                        missing_files.append(file)

                if missing_files:
                    print("\n以下文件未找到:")
                    for file in missing_files:
                        print(f"- {file}")
                    raise Exception(f"缺少必要文件: {', '.join(missing_files)}")

                # 3. 读取CFD结果
                cl, cd, cm = self.read_output_values()
                print(f"\nCFD计算结果: Cl={cl:.6f}, Cd={cd:.6f}, Cm={cm:.6f}")

                # 4. 更新数据文件
                self.update_data_files(cl, cd, cm)

                return cl, cd, cm

            finally:
                # 确保返回到原始工作目录
                os.chdir(original_dir)

        except Exception as e:
            print(f"\nCFD工作流执行失败: {str(e)}")
            raise


def run_cfd_workflow(point: np.ndarray) -> Tuple[float, float, float]:
    """
    运行CFD工作流的便捷函数

    Parameters:
    -----------
    point: np.ndarray
        包含Chord, Distance, Fai值的数组

    Returns:
    --------
    Tuple[float, float, float]
        计算得到的Cl, Cd, Cm值
    """
    cfd = CFDAutomation()
    return cfd.run_cfd_workflow(point)